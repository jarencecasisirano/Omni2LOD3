# 04_json_fix.py
"""
CityJSON Empty Geometry Fixer
------------------------------
Removes empty geometry entries (boundaries: []) from CityJSON files 
to fix val3dity error 902 - EMPTY_PRIMITIVE

Author: For LOD2 Building Model Validation Pipeline
Date: 2026-01-28
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import shutil


class CityJSONFixer:
    """Fix CityJSON files by removing empty geometries."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_geometries_removed': 0,
            'total_objects_checked': 0
        }
    
    def find_json_files(self) -> List[Path]:
        """Find all .json files in input directory."""
        if not self.input_dir.exists():
            print(f"❌ Input path not found: {self.input_dir}")
            return []
        
        # Check if it's a file (user gave specific file path)
        if self.input_dir.is_file():
            if self.input_dir.suffix.lower() == '.json':
                print(f"ℹ️  Single file provided: {self.input_dir.name}")
                self.stats['total_files'] = 1
                return [self.input_dir]
            else:
                print(f"❌ Not a JSON file: {self.input_dir}")
                return []
        
        # It's a directory - find all JSON files
        json_files = list(self.input_dir.glob("*.json"))
        self.stats['total_files'] = len(json_files)
        
        if len(json_files) == 0:
            print(f"❌ No JSON files found in: {self.input_dir}")
        
        return sorted(json_files)
    
    def check_geometry_empty(self, geometry: Dict) -> bool:
        """
        Check if a geometry entry is empty.
        
        A geometry is considered empty if:
        - boundaries key is missing, OR
        - boundaries is an empty list []
        
        Args:
            geometry: Geometry dictionary from CityJSON
        
        Returns:
            True if empty, False if has data
        """
        if 'boundaries' not in geometry:
            return True
        
        boundaries = geometry['boundaries']
        
        # Check if empty list or None
        if boundaries is None or len(boundaries) == 0:
            return True
        
        return False
    
    def fix_city_object(self, city_object: Dict, object_id: str) -> Tuple[Dict, int]:
        """
        Remove empty geometries from a single CityObject.
        
        Args:
            city_object: CityObject dictionary
            object_id: ID of the CityObject
        
        Returns:
            Tuple of (fixed_object, num_removed)
        """
        removed_count = 0
        
        if 'geometry' not in city_object:
            return city_object, 0
        
        original_geometries = city_object['geometry']
        valid_geometries = []
        
        for i, geom in enumerate(original_geometries):
            if self.check_geometry_empty(geom):
                geom_type = geom.get('type', 'Unknown')
                lod = geom.get('lod', 'N/A')
                print(f"    ├─ Removing empty {geom_type} (LOD {lod})")
                removed_count += 1
            else:
                valid_geometries.append(geom)
        
        # Update geometry array
        city_object['geometry'] = valid_geometries
        
        return city_object, removed_count
    
    def fix_cityjson_file(self, file_path: Path) -> Tuple[bool, Dict]:
        """
        Fix a single CityJSON file.
        
        Args:
            file_path: Path to input file
        
        Returns:
            Tuple of (success, report_dict)
        """
        print(f"\n📄 Processing: {file_path.name}")
        
        try:
            # Load JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate it's CityJSON
            if 'CityObjects' not in data:
                print(f"  ⚠️  Not a valid CityJSON file (missing 'CityObjects')")
                return False, {'error': 'Invalid CityJSON structure'}
            
            # Process each CityObject
            city_objects = data['CityObjects']
            total_removed = 0
            objects_modified = 0
            
            for obj_id, obj_data in city_objects.items():
                self.stats['total_objects_checked'] += 1
                
                fixed_obj, removed = self.fix_city_object(obj_data, obj_id)
                
                if removed > 0:
                    objects_modified += 1
                    total_removed += removed
                    print(f"  ✓ Fixed CityObject '{obj_id}' - removed {removed} empty geometries")
                
                # Update in place
                city_objects[obj_id] = fixed_obj
            
            # Create output directory if needed
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine output filename
            if file_path.parent == self.input_dir or self.input_dir.is_file():
                # Input was a directory or single file - use same filename
                output_filename = file_path.name
            else:
                # Nested structure - preserve it
                output_filename = file_path.name
            
            # Add "_FIXED" suffix before extension
            output_name = file_path.stem + "_FIXED" + file_path.suffix
            output_path = self.output_dir / output_name
            
            # Save fixed file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self.stats['processed_files'] += 1
            self.stats['total_geometries_removed'] += total_removed
            
            report = {
                'input_file': str(file_path),
                'output_file': str(output_path),
                'objects_modified': objects_modified,
                'geometries_removed': total_removed,
                'status': 'success'
            }
            
            if total_removed > 0:
                print(f"  ✅ Saved to: {output_path.name}")
                print(f"  📊 Summary: {objects_modified} objects modified, {total_removed} geometries removed")
            else:
                print(f"  ℹ️  No empty geometries found - file saved as: {output_path.name}")
            
            return True, report
            
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON decode error: {e}")
            return False, {'error': f'JSON decode error: {e}'}
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return False, {'error': f'Unexpected error: {e}'}
    
    def interactive_mode(self):
        """Interactive mode: let user select which files to fix."""
        print("\n" + "="*70)
        print("🔧 CityJSON Empty Geometry Fixer")
        print("="*70)
        print(f"\nInput directory:  {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Find files
        json_files = self.find_json_files()
        
        if not json_files:
            print("\n❌ No JSON files found!")
            return
        
        print(f"\n📁 Found {len(json_files)} JSON file(s):")
        print()
        
        for i, file in enumerate(json_files, 1):
            file_size = file.stat().st_size / 1024  # KB
            print(f"  [{i}] {file.name:<40} ({file_size:.1f} KB)")
        
        print(f"  [{len(json_files) + 1}] Process ALL files")
        print(f"  [0] Exit")
        
        # Get user choice
        while True:
            try:
                choice = input(f"\n👉 Select file to fix [0-{len(json_files) + 1}]: ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    print("\n👋 Exiting...")
                    return
                
                elif choice_num == len(json_files) + 1:
                    # Process all
                    print("\n🚀 Processing ALL files...")
                    reports = []
                    
                    for file in json_files:
                        success, report = self.fix_cityjson_file(file)
                        reports.append(report)
                    
                    self.print_summary()
                    break
                
                elif 1 <= choice_num <= len(json_files):
                    # Process single file
                    selected_file = json_files[choice_num - 1]
                    success, report = self.fix_cityjson_file(selected_file)
                    
                    self.print_summary()
                    
                    # Ask if want to continue
                    again = input("\n✨ Fix another file? (y/n): ").strip().lower()
                    if again != 'y':
                        break
                    
                    # Reset stats for next file
                    self.stats = {
                        'total_files': len(json_files),
                        'processed_files': 0,
                        'total_geometries_removed': 0,
                        'total_objects_checked': 0
                    }
                
                else:
                    print(f"⚠️  Please enter a number between 0 and {len(json_files) + 1}")
                    
            except ValueError:
                print("⚠️  Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user")
                return
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*70)
        print("📊 PROCESSING SUMMARY")
        print("="*70)
        print(f"Files processed:         {self.stats['processed_files']}/{self.stats['total_files']}")
        print(f"CityObjects checked:     {self.stats['total_objects_checked']}")
        print(f"Empty geometries removed: {self.stats['total_geometries_removed']}")
        print("="*70)
        
        if self.stats['total_geometries_removed'] > 0:
            print("\n✅ Success! Your fixed files are ready for val3dity validation.")
            print(f"📂 Check: {self.output_dir}")
        else:
            print("\n✨ All files were already clean (no empty geometries found).")


def main():
    """Main entry point."""
    # Default paths (Windows-style, will work on Windows)
    default_input = r"C:\Projects\Thesis\data\json_model"
    default_output = r"C:\Projects\Thesis\outputs\LOD2_json"
    
    print("\n" + "="*70)
    print("🏗️  LOD2 CityJSON Geometry Fixer")
    print("="*70)
    print("\nThis script removes empty geometry entries that cause:")
    print("  • val3dity Error 902: EMPTY_PRIMITIVE")
    print("  • 'empty Solid, contains no points and/or surfaces'")
    print("\n" + "─"*70)
    print("📂 INPUT: Provide a directory containing JSON files")
    print("   Example: C:\\Projects\\Thesis\\data\\json_model")
    print("   (or press Enter to use default)")
    print("─"*70)
    
    # Allow user to override paths
    input_dir = input(f"\nInput directory [{default_input}]: ").strip() or default_input
    output_dir = input(f"Output directory [{default_output}]: ").strip() or default_output
    
    # Create fixer and run
    fixer = CityJSONFixer(input_dir, output_dir)
    fixer.interactive_mode()
    
    print("\n✨ Done!\n")


if __name__ == "__main__":
    main()