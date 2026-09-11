import gmsh
import sys
import math
import random
import json
import os
import argparse

class MicrowaveGeometryGenerator:
    def __init__(self, variation_level="LOW"):
        """
        Initialize the generator.
        variation_level: "LOW", "MEDIUM", "HIGH" controls the randomness of the geometry.
        """
        self.variation = variation_level
        self.model_name = "microwave_sim"
        self.tags = {
            "volumes": {},
            "surfaces": {}
        }
        self.rotation_center = [0.0, 0.0, 0.0]
        
        # Default Parameters (in meters)
        self.params = {
            "cavity_width": 0.35,
            "cavity_depth": 0.35,
            "cavity_height": 0.25,
            "wg_width": 0.086,  # WR340 or similar
            "wg_height": 0.043,
            "wg_length": 0.10,
            "bowl_radius": 0.08,
            "bowl_height": 0.02,
            "dumpling_radius_skin": 0.015,
            "dumpling_radius_filling": 0.010
        }

    def _generate_random_parameters(self):
        """Adjust parameters based on variation level."""
        if self.variation == "LOW":
            return
        
        factor = 0.1 if self.variation == "MEDIUM" else 0.2
        
        # Randomize cavity slightly
        self.params["cavity_width"] *= (1 + random.uniform(-factor, factor))
        
        # Randomize bowl position slightly (will be handled in placement)

    def build_geometry(self):
        if not gmsh.isInitialized():
            gmsh.initialize()
            
        gmsh.model.add(self.model_name)
        
        self._generate_random_parameters()
        
        p = self.params
        
        # ---------------------------------------------------------
        # 1. Create Air Domain (Cavity + Waveguide)
        # ---------------------------------------------------------
        # Cavity Box
        cavity = gmsh.model.occ.addBox(0, 0, 0, p["cavity_width"], p["cavity_depth"], p["cavity_height"])
        
        # Waveguide (attached to the right side for example)
        wg_x = p["cavity_width"]
        wg_y = (p["cavity_depth"] - p["wg_width"]) / 2
        wg_z = (p["cavity_height"] - p["wg_height"]) / 2
        waveguide = gmsh.model.occ.addBox(wg_x, wg_y, wg_z, p["wg_length"], p["wg_width"], p["wg_height"])
        
        # Fuse Cavity and Waveguide to create the main Air body
        air_dimtags, _ = gmsh.model.occ.fuse([(3, cavity)], [(3, waveguide)])
        main_air_tag = air_dimtags[0][1]
        
        # ---------------------------------------------------------
        # 2. Create Stirrer (Metallic - will be holes in Air or Wall surfaces)
        # ---------------------------------------------------------
        # Simple fan shape on the ceiling
        stirrer_center = [p["cavity_width"]/2, p["cavity_depth"]/2, p["cavity_height"] - 0.02]
        stirrer_blade_len = 0.08
        stirrer_blade_width = 0.02
        
        stirrer_parts = []
        # Hub
        hub = gmsh.model.occ.addCylinder(stirrer_center[0], stirrer_center[1], stirrer_center[2], 
                                         0, 0, 0.01, 0.01)
        stirrer_parts.append((3, hub))
        
        # Blades (Random rotation based on variation)
        if self.variation == "LOW":
            num_blades = 2
            start_angle = 0.0
        else:
            num_blades = random.choice([2, 3, 4])
            start_angle = random.uniform(0, 2*math.pi)
        
        for i in range(num_blades):
            angle = start_angle + (i * 2 * math.pi / num_blades)
            # Create blade using box and rotate
            blade = gmsh.model.occ.addBox(-stirrer_blade_width/2, 0, 0, 
                                          stirrer_blade_width, stirrer_blade_len, 0.005)
            
            # Rotate blade to position
            gmsh.model.occ.rotate([(3, blade)], 0, 0, 0, 0, 0, 1, angle) # Rotate around Z local
            gmsh.model.occ.translate([(3, blade)], stirrer_center[0], stirrer_center[1], stirrer_center[2])
            stirrer_parts.append((3, blade))
            
        # Fuse stirrer parts
        stirrer_dimtags, _ = gmsh.model.occ.fuse([stirrer_parts[0]], stirrer_parts[1:])
        stirrer_tag = stirrer_dimtags[0][1]

        # ---------------------------------------------------------
        # 3. Create Bowl (Dielectric)
        # ---------------------------------------------------------
        bowl_center_x = p["cavity_width"] / 2
        bowl_center_y = p["cavity_depth"] / 2
        bowl_z = 0.01 # Slightly elevated from floor
        
        bowl = gmsh.model.occ.addCylinder(bowl_center_x, bowl_center_y, bowl_z, 
                                          0, 0, p["bowl_height"], p["bowl_radius"])
        
        # Set Rotation Center (for output file)
        self.rotation_center = [bowl_center_x, bowl_center_y, bowl_z]

        # ---------------------------------------------------------
        # 4. Create Food (Dumplings: Skin + Filling)
        # ---------------------------------------------------------
        dumplings_skin = []
        dumplings_filling = []
        
        if self.variation == "LOW":
            num_dumplings = 3
        else:
            num_dumplings = random.randint(2, 5)
        
        for i in range(num_dumplings):
            # Random position on the bowl
            r = random.uniform(0, p["bowl_radius"] - p["dumpling_radius_skin"])
            theta = random.uniform(0, 2*math.pi)
            dx = r * math.cos(theta)
            dy = r * math.sin(theta)
            
            d_x = bowl_center_x + dx
            d_y = bowl_center_y + dy
            d_z = bowl_z + p["bowl_height"] + p["dumpling_radius_skin"] # Sit on top of bowl
            
            # Outer Sphere (Skin)
            skin = gmsh.model.occ.addSphere(d_x, d_y, d_z, p["dumpling_radius_skin"])
            # Inner Sphere (Filling)
            filling = gmsh.model.occ.addSphere(d_x, d_y, d_z, p["dumpling_radius_filling"])
            
            dumplings_skin.append((3, skin))
            dumplings_filling.append((3, filling))

        # ---------------------------------------------------------
        # 5. Boolean Fragment (The most important part)
        # ---------------------------------------------------------
        # We need to fragment Air with (Stirrer, Bowl, Dumplings).
        # Note: Dumpling Skin and Filling overlap. We must handle this.
        # Strategy: 
        # 1. Fragment Skin and Filling first to get (Skin_Shell, Filling_Core).
        # 2. Fragment Air with everything else.
        
        # List of all object dimtags to fragment into the air
        objects_to_fragment = []
        objects_to_fragment.append((3, stirrer_tag))
        objects_to_fragment.append((3, bowl))
        
        # IMPORTANT: Interleave Skin and Filling for the loop below to work correctly
        # Order: Stirrer, Bowl, Skin1, Filling1, Skin2, Filling2, ...
        for s, f in zip(dumplings_skin, dumplings_filling):
            objects_to_fragment.append(s)
            objects_to_fragment.append(f)
        
        # Perform Fragment: Air cut by Objects
        # object_map maps input tags to output tags.
        out_dimtags, out_map = gmsh.model.occ.fragment([(3, main_air_tag)], objects_to_fragment)
        
        gmsh.model.occ.synchronize()
        
        # ---------------------------------------------------------
        # 6. Identify and Assign Physical Groups
        # ---------------------------------------------------------
        # out_map structure:
        # out_map[0] -> fragments of Air
        # out_map[1] -> fragments of Stirrer
        # out_map[2] -> fragments of Bowl
        # out_map[3] -> fragments of Skin 1
        # out_map[4] -> fragments of Filling 1
        # out_map[5] -> fragments of Skin 2
        # out_map[6] -> fragments of Filling 2
        # ...
        
        # Helper to flatten list of tags
        def get_tags(dimtags):
            return [t[1] for t in dimtags]

        # 1. Air
        # The air volume is whatever remains of the original air box, excluding the volumes occupied by other objects.
        all_object_tags = []
        for i in range(1, len(out_map)):
            all_object_tags.extend(get_tags(out_map[i]))
        
        air_tags = []
        for tag in get_tags(out_map[0]):
            if tag not in all_object_tags:
                air_tags.append(tag)
                
        # 2. Stirrer (Metal)
        stirrer_vol_tags = get_tags(out_map[1])
        
        # 3. Bowl
        bowl_vol_tags = get_tags(out_map[2])
        
        # 4. Dumplings
        skin_vol_tags = []
        filling_vol_tags = []
        
        idx = 3
        for i in range(num_dumplings):
            # Skin input was index `idx`
            # Filling input was index `idx+1`
            
            s_tags = get_tags(out_map[idx])     # Skin fragments
            f_tags = get_tags(out_map[idx+1])   # Filling fragments
            
            # The filling is the intersection (present in both because Filling is inside Skin)
            current_filling = [t for t in f_tags if t in s_tags]
            # The skin shell is the rest of skin
            current_skin = [t for t in s_tags if t not in current_filling]
            
            filling_vol_tags.extend(current_filling)
            skin_vol_tags.extend(current_skin)
            
            idx += 2

        # Create Physical Volumes
        pg_air = gmsh.model.addPhysicalGroup(3, air_tags, name="air")
        
        # Stirrer: Added Physical Group as requested
        pg_stirrer = gmsh.model.addPhysicalGroup(3, stirrer_vol_tags, name="stirrer")
        
        # Bowl: Assign to "bowl" AND "glass" for compatibility
        pg_bowl = gmsh.model.addPhysicalGroup(3, bowl_vol_tags, name="bowl")
        gmsh.model.addPhysicalGroup(3, bowl_vol_tags, name="glass")
        
        pg_skin = gmsh.model.addPhysicalGroup(3, skin_vol_tags, name="mass_skin")
        pg_filling = gmsh.model.addPhysicalGroup(3, filling_vol_tags, name="mass_filling")
        
        # Store for JSON
        self.tags["volumes"] = {
            "air": pg_air,
            "stirrer": pg_stirrer,
            "bowl": pg_bowl,
            "mass_skin": pg_skin,
            "mass_filling": pg_filling
        }

        # ---------------------------------------------------------
        # 7. Surfaces (Port and Walls)
        # ---------------------------------------------------------
        # Port: The face of the waveguide at max X
        air_boundaries = gmsh.model.getBoundary([(3, t) for t in air_tags], combined=True, oriented=False)
        
        port_surfaces = []
        wall_surfaces = []
        
        wg_end_x = p["cavity_width"] + p["wg_length"]
        
        for dim, tag in air_boundaries:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            if math.isclose(com[0], wg_end_x, abs_tol=1e-4):
                port_surfaces.append(tag)
            else:
                wall_surfaces.append(tag)
        
        # Also, the Stirrer surface is a wall (Metal boundary inside air)
        stirrer_boundaries = gmsh.model.getBoundary([(3, t) for t in stirrer_vol_tags], combined=True, oriented=False)
        for dim, tag in stirrer_boundaries:
            wall_surfaces.append(tag)

        # Create Physical Surfaces
        pg_port = gmsh.model.addPhysicalGroup(2, port_surfaces, name="port")
        pg_wall = gmsh.model.addPhysicalGroup(2, wall_surfaces, name="wall")
        
        self.tags["surfaces"] = {
            "port": pg_port,
            "wall": pg_wall
        }

    def generate_mesh(self, filename_prefix="microwave_mesh"):
        # Mesh Settings
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.03)
        
        gmsh.model.mesh.generate(3)
        
        # 1. Save .msh
        msh_filename = f"{filename_prefix}.msh"
        gmsh.write(msh_filename)
        
        # 2. Save .json
        json_data = {
            "geometry_parameters": self.params,
            "physical_groups": {
                "air": self.tags["volumes"]["air"],
                "stirrer": self.tags["volumes"]["stirrer"],
                "bowl": self.tags["volumes"]["bowl"],
                "mass_skin": self.tags["volumes"]["mass_skin"],
                "mass_filling": self.tags["volumes"]["mass_filling"],
                "port": self.tags["surfaces"]["port"],
                "wall": self.tags["surfaces"]["wall"]
            },
            "materials": {
                "air": {"epsilon": "1.0"},
                "bowl": {"epsilon": "5.5"},
                "mass_skin": {"epsilon": "25 - 5j"},
                "mass_filling": {"epsilon": "45 - 15j"}
            }
        }
        
        json_filename = f"{filename_prefix}.json"
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=4)
            
        # 3. Save rotation_center.txt
        txt_filename = "rotation_center.txt"
        with open(txt_filename, 'w') as f:
            f.write(f"{self.rotation_center[0]:.6f} {self.rotation_center[1]:.6f} {self.rotation_center[2]:.6f}")
            
        print(f"Successfully generated:\n- {msh_filename}\n- {json_filename}\n- {txt_filename}")
        gmsh.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random microwave geometry.")
    parser.add_argument("--variation", choices=["LOW", "MEDIUM", "HIGH"], default="LOW", 
                        help="Level of randomness for the geometry.")
    args = parser.parse_args()
    
    print(f"Generating geometry with variation level: {args.variation}")
    
    generator = MicrowaveGeometryGenerator(variation_level=args.variation)
    generator.build_geometry()
    generator.generate_mesh("microwave_sim")
