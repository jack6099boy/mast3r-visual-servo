import os
import sys
from pathlib import Path

# Add mast3r-research to sys.path for MASt3R imports
sys.path.insert(0, str(Path(__file__).parent.parent / "mast3r-research"))

from template_system import TemplateManager, TemplateMatcher

def demo_manager():
    print("=== TemplateManager Demo ===")
    manager = TemplateManager()
    rules = manager.list_rules()
    print(f"Available rules: {rules}")
    
    # Ensure a_lab/dog exists
    if 'a_lab' not in rules:
        print("Creating a_lab rule...")
        manager.create_key('a_lab', 'dog')
        rules = manager.list_rules()
    
    templates = manager.load_templates('a_lab', 'dog')
    print(f"Loaded {len(templates)} templates")
    for t in templates:
        print(f"  - {t.img_path.name} (ROI loaded)")

def demo_gui():
    print("\n=== ROIAnnotator GUI Demo ===")
    print("Instructions: Select rule/key/image (front.jpg should show if no prior ROI),")
    print("click to draw polygon points, Clear Points, Save ROI. Close window to continue.")
    manager = TemplateManager()
    annotator = ROIAnnotator(manager)
    annotator.launch()  # This will block until GUI is closed

def demo_matcher():
    print("\n=== TemplateMatcher Demo ===")
    manager = TemplateManager()
    target_path = Path(__file__).parent.parent / 'mast3r-research' / 'assets' / 'demo.jpg'
    print(f"Using target: {target_path}")
    
    matcher = TemplateMatcher(manager)
    result = matcher.match('a_lab', 'dog', target_path)
    print("Results:")
    print(f"Best match: {result['best']}")
    print(f"All matches: {result['all']}")
    
    best_score = result['best']['score'] if result['best'] else 0
    if best_score > 0:
        print("✅ Matching score > 0 (MASt3R working)")
    else:
        print("ℹ️ Score = 0 (using dummy mode, install mast3r-research for real matching)")

def run_tests():
    print("\n=== Unit Tests ===")
    manager = TemplateManager()
    assert 'a_lab' in manager.list_rules(), "Rule 'a_lab' not found"
    templates = manager.load_templates('a_lab', 'dog')
    assert len(templates) >= 1, "No templates loaded"
    print("✅ All tests passed!")

if __name__ == '__main__':
    demo_manager()
    run_tests()
    demo_matcher()
    # demo_gui()  # Skip GUI: tkinter not available in env; run manually if needed