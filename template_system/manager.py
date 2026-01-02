from dataclasses import dataclass
from typing import List, Dict, Any
import json
from pathlib import Path

@dataclass
class Template:
    img_path: Path
    roi_path: Path
    roi_data: Dict[str, Any]  # 載入後的 JSON

class TemplateManager:
    """管理 templates 資料夾結構"""
    def __init__(self, root_dir: Path = Path('templates')):
        self.root: Path = root_dir
        self.root.mkdir(exist_ok=True)
    
    def list_rules(self) -> List[str]:
        """列出所有 rule names"""
        if not self.root.exists():
            return []
        return [d.name for d in self.root.iterdir() if d.is_dir()]
    
    def list_keys(self, rule: str) -> List[str]:
        """列出指定 rule 下的 key names"""
        rule_dir = self.root / rule
        if not rule_dir.exists():
            return []
        return [d.name for d in rule_dir.iterdir() if d.is_dir()]
    
    def load_templates(self, rule: str, key: str) -> List[Template]:
        """載入指定 key 下所有 templates (jpg/jpeg + *_roi.json)"""
        key_dir = self.root / rule / key
        if not key_dir.exists():
            return []
        templates: List[Template] = []
        # 支援 .jpg 和 .jpeg 副檔名
        for img_path in key_dir.iterdir():
            if img_path.suffix.lower() not in ('.jpg', '.jpeg'):
                continue
            roi_filename = f"{img_path.stem}_roi.json"
            roi_path = key_dir / roi_filename
            if roi_path.exists():
                try:
                    with open(roi_path, 'r', encoding='utf-8') as f:
                        roi_data = json.load(f)
                    templates.append(Template(img_path, roi_path, roi_data))
                except json.JSONDecodeError:
                    print(f"Invalid JSON in {roi_path}")
        return templates
    
    def create_key(self, rule: str, key: str) -> Path:
        """創建 key 資料夾"""
        key_dir = self.root / rule / key
        key_dir.mkdir(parents=True, exist_ok=True)
        return key_dir