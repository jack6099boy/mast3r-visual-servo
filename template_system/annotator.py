import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
from typing import List
from pathlib import Path
from .manager import TemplateManager

class ROIAnnotator:
    """GUI 標注工具"""
    def __init__(self, manager: TemplateManager):
        self.manager = manager
        self.root: tk.Tk | None = None
        self.canvas: tk.Canvas | None = None
        self.img_path: Path | None = None
        self.coord_label: tk.Label | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.points: List[tuple[float, float]] = []
        self.edit_mode: bool = False
        self.selected_index: int | None = None
        self.point_ovals: List[int] = []
        self.status_label: tk.Label | None = None
        self.scale: float = 1.0
        self.img_width: int = 0
        self.img_height: int = 0

    def launch(self):
        """啟動 GUI 介面"""
        self.root = tk.Tk()
        self.root.title("ROI Annotator - Template System")
        self.root.geometry("900x700")

        # Rule 選擇
        tk.Label(self.root, text="Rule:", font=("Arial", 12)).pack(pady=5)
        self.rule_var = tk.StringVar()
        self.rule_combo = ttk.Combobox(self.root, textvariable=self.rule_var, width=30)
        self.rule_combo.pack(pady=5)
        self.rule_combo.bind('<<ComboboxSelected>>', self.on_rule_select)

        # Key 選擇
        tk.Label(self.root, text="Key:", font=("Arial", 12)).pack(pady=5)
        self.key_var = tk.StringVar()
        self.key_combo = ttk.Combobox(self.root, textvariable=self.key_var, width=30)
        self.key_combo.pack(pady=5)
        self.key_combo.bind('<<ComboboxSelected>>', self.on_key_select)

        # Image 選擇 (僅無 ROI 的 jpg)
        tk.Label(self.root, text="Image:", font=("Arial", 12)).pack(pady=5)
        self.img_var = tk.StringVar()
        self.img_combo = ttk.Combobox(self.root, textvariable=self.img_var, width=30)
        self.img_combo.pack(pady=5)
        self.img_combo.bind('<<ComboboxSelected>>', self.on_img_select)

        # Canvas 框架
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg='white', width=800, height=500)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # 說明
        tk.Label(self.root, text="新繪模式：左鍵添加點；編輯模式：左鍵選點拖拉/添加、雙擊刪點、右鍵刪點。Clear 清空，Save 儲存。",
                 font=("Arial", 10), fg="blue").pack(pady=5)

        self.status_label = tk.Label(self.root, text="狀態: 新繪模式 0 點", font=("Arial", 10), fg="purple")
        self.status_label.pack(pady=5)
        self.coord_label = tk.Label(self.root, text="座標: (0, 0)", font=("Arial", 10), fg="green")
        self.coord_label.pack(pady=5)

        # 按鈕
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Clear Points", command=self.clear_points, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save ROI", command=self.save_roi, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Refresh Rules", command=self.update_rules, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Toggle Edit", command=self.toggle_edit, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Quit", command=self.root.quit, width=12).pack(side=tk.LEFT, padx=5)

        self.update_rules()
        self.root.mainloop()

    def update_rules(self):
        rules = self.manager.list_rules()
        self.rule_combo['values'] = rules if rules else ["(無 rules)"]

    def on_rule_select(self, event=None):
        rule = self.rule_var.get()
        if rule:
            keys = self.manager.list_keys(rule)
            self.key_combo['values'] = keys if keys else ["(無 keys)"]
            self.key_var.set("")
            self.img_combo['values'] = []

    def on_key_select(self, event=None):
        rule = self.rule_var.get()
        key = self.key_var.get()
        if rule and key:
            key_dir = self.manager.root / rule / key
            # 支援 .jpg 和 .jpeg 副檔名
            images = [p.name for p in key_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg')]
            self.img_combo['values'] = images if images else ["(無未標注影像)"]
            self.img_var.set("")

    def on_img_select(self, event=None):
        rule = self.rule_var.get()
        key = self.key_var.get()
        img_name = self.img_var.get()
        if rule and key and img_name:
            self.img_path = self.manager.root / rule / key / img_name
            self.load_image_and_roi()

    def load_image(self):
        self.clear_points()
        try:
            img = Image.open(self.img_path)
            self.img_width, self.img_height = img.size

            # 調整大小適合 canvas (max 800x500)
            canvas_w, canvas_h = 800, 500
            ratio = min(canvas_w / self.img_width, canvas_h / self.img_height)
            new_w = int(self.img_width * ratio)
            new_h = int(self.img_height * ratio)
            self.scale = ratio

            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_resized)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        except Exception as e:
            messagebox.showerror("錯誤", f"載入影像失敗: {e}")

    def on_click(self, event):
        if not self.photo:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        closest = self.find_closest_point(canvas_x, canvas_y)
        if self.edit_mode:
            if closest is not None:
                self.selected_index = closest
                return
            else:
                # 添加新點
                x = canvas_x / self.scale
                y = canvas_y / self.scale
                self.points.append((x, y))
                self.selected_index = len(self.points) - 1
                self.draw_polygon()
                self.update_status()
        else:
            # 新繪模式：添加點
            x = canvas_x / self.scale
            y = canvas_y / self.scale
            self.points.append((x, y))
            self.draw_polygon()
            self.update_status()

    def on_mouse_move(self, event):
        if self.photo and self.scale > 0:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            x = canvas_x / self.scale
            y = canvas_y / self.scale
            self.coord_label.config(text=f"座標: ({int(x)}, {int(y)})")
        else:
            self.coord_label.config(text="座標: (0, 0)")

    def clear_points(self):
        self.points = []
        self.selected_index = None
        self.canvas.delete("polygon")
        if self.photo:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.update_status()

    def save_roi(self):
        if not self.points or len(self.points) < 3:
            messagebox.showerror("錯誤", "至少需要 3 個點來形成多邊形")
            return
        if self.points[0] != self.points[-1]:
            self.points.append(self.points[0])  # 閉合多邊形

        roi_data = {
            "version": "1.0",
            "image_metadata": {
                "width": self.img_width,
                "height": self.img_height,
                "filename": self.img_path.name
            },
            "polygons": [
                {
                    "label": "target",
                    "points": [[float(p[0]), float(p[1])] for p in self.points],
                    "metadata": {
                        "annotated_by": "user",
                        "timestamp": "2024-01-01T00:00:00Z"  # 可替換為實際時間
                    }
                }
            ]
        }

        roi_filename = f"{self.img_path.stem}_roi.json"
        roi_path = self.img_path.parent / roi_filename
        try:
            with open(roi_path, 'w', encoding='utf-8') as f:
                json.dump(roi_data, f, indent=2)
            messagebox.showinfo("成功", f"ROI 已儲存至 {roi_filename}")
            self.clear_points()
            # Refresh images
            self.on_key_select()
        except Exception as e:
            messagebox.showerror("錯誤", f"儲存失敗: {e}")


    def load_image_and_roi(self):
        self.load_image()
        self.load_roi_if_exists()

    def load_roi_if_exists(self):
        if not self.img_path:
            return
        roi_filename = f"{self.img_path.stem}_roi.json"
        roi_path = self.img_path.parent / roi_filename
        if roi_path.exists():
            try:
                with open(roi_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get('polygons') and len(data['polygons']) > 0:
                    poly_points = data['polygons'][0].get('points', [])
                    if len(poly_points) > 2:
                        self.points = [(float(p[0]), float(p[1])) for p in poly_points[:-1]]
                        self.edit_mode = True
                        self.draw_polygon()
                        self.update_status()
            except Exception as e:
                print(f"載入 ROI 失敗: {e}")

    def draw_polygon(self):
        self.canvas.delete("polygon")
        self.point_ovals = []
        if not self.points or len(self.points) < 2:
            return
        scaled_points = [(x * self.scale, y * self.scale) for x, y in self.points]
        # 繪製線條 (閉合)
        n = len(scaled_points)
        for i in range(n):
            j = (i + 1) % n
            self.canvas.create_line(scaled_points[i][0], scaled_points[i][1],
                                    scaled_points[j][0], scaled_points[j][1],
                                    fill='red', width=2, tags="polygon")
        # 繪製點
        for sx, sy in scaled_points:
            oval = self.canvas.create_oval(sx-5, sy-5, sx+5, sy+5,
                                           fill='red', outline='blue', tags="polygon")
            self.point_ovals.append(oval)

    def update_status(self):
        n_points = len(self.points)
        mode_str = "編輯" if self.edit_mode else "新繪"
        status_text = f"狀態: {mode_str}模式 {n_points} 點"
        if self.status_label:
            self.status_label.config(text=status_text)

    def toggle_edit(self):
        self.edit_mode = not self.edit_mode
        if not self.edit_mode:
            self.clear_points()
        else:
            self.load_roi_if_exists()
        self.update_status()

    def find_closest_point(self, canvas_x: float, canvas_y: float, threshold: float = 10.0) -> int | None:
        min_dist = float('inf')
        closest_i = None
        for i, (px, py) in enumerate(self.points):
            sx = px * self.scale
            sy = py * self.scale
            dist = ((sx - canvas_x) ** 2 + (sy - canvas_y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_i = i
        if min_dist < threshold:
            return closest_i
        return None

    def on_drag(self, event):
        if self.edit_mode and self.selected_index is not None and self.photo:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            x = canvas_x / self.scale
            y = canvas_y / self.scale
            self.points[self.selected_index] = (x, y)
            self.draw_polygon()

    def on_double_click(self, event):
        if not self.edit_mode or not self.photo or not self.points:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        closest = self.find_closest_point(canvas_x, canvas_y)
        if closest is not None and len(self.points) > 3:
            del self.points[closest]
            self.selected_index = None
            self.draw_polygon()
            self.update_status()

    def on_right_click(self, event):
        if not self.edit_mode or not self.photo or not self.points:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        closest = self.find_closest_point(canvas_x, canvas_y)
        if closest is not None and len(self.points) > 3:
            del self.points[closest]
            self.selected_index = None
            self.draw_polygon()
            self.update_status()


def main():
    """啟動 ROI 標注工具"""
    manager = TemplateManager()
    app = ROIAnnotator(manager)
    app.launch()


if __name__ == "__main__":
    main()
