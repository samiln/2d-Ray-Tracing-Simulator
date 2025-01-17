import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, colorchooser
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray
    intensity: float = 1.0
    color: str = 'red'


@dataclass
class Surface:
    start: np.ndarray
    end: np.ndarray
    refractive_index: float = 1.0
    is_mirror: bool = False


class RayTracer:
    def __init__(self):
        self.surfaces: List[Surface] = []
        self.rays: List[List[Ray]] = []
        self.max_depth = 5
        self.min_intensity = 0.01

    def clear_rays(self):
        self.rays = []

    def trace_ray(self, ray: Ray, depth: int = 0) -> List[Ray]:
        if depth >= self.max_depth or ray.intensity < self.min_intensity:
            return [ray]

        closest_intersection = None
        closest_surface = None
        min_distance = float('inf')

        for surface in self.surfaces:
            intersection = self._find_intersection(ray, surface)
            if intersection is not None:
                distance = np.linalg.norm(intersection - ray.origin)
                if distance < min_distance:
                    min_distance = distance
                    closest_intersection = intersection
                    closest_surface = surface

        if closest_intersection is None:
            return [ray]

        ray_path = [ray]
        reflected, refracted = self._reflect_and_refract(
            ray, closest_surface, closest_intersection)
        if reflected:
            ray_path.extend(self.trace_ray(reflected, depth + 1))
        if refracted:
            ray_path.extend(self.trace_ray(refracted, depth + 1))

        return ray_path

    def _find_intersection(self, ray: Ray, surface: Surface) -> Optional[np.ndarray]:
        def cross2d(a, b):
            return a[0] * b[1] - a[1] * b[0]

        v1 = ray.direction
        v2 = surface.end - surface.start
        v3 = surface.start - ray.origin

        cross = cross2d(v1, v2)
        if abs(cross) < 1e-10:
            return None

        t1 = cross2d(v3, v2) / cross
        t2 = cross2d(v3, v1) / cross

        if t1 >= 0 and 0 <= t2 <= 1:
            return ray.origin + t1 * ray.direction

        return None

    def _reflect_and_refract(self, ray: Ray, surface: Surface,
                             intersection: np.ndarray) -> Tuple[Optional[Ray], Optional[Ray]]:
        surface_dir = surface.end - surface.start
        normal = np.array([-surface_dir[1], surface_dir[0]])
        normal = normal / np.linalg.norm(normal)

        if np.dot(normal, ray.direction) > 0:
            normal = -normal
            n1, n2 = surface.refractive_index, 1.0
        else:
            n1, n2 = 1.0, surface.refractive_index

        cos_theta1 = -np.dot(normal, ray.direction)
        sin_theta1 = np.sqrt(1 - cos_theta1 * cos_theta1)
        sin_theta2 = n1 / n2 * sin_theta1

        if sin_theta2 > 1:  # Total internal reflection
            reflection = ray.direction - 2 * np.dot(ray.direction, normal) * normal
            return Ray(intersection, reflection, ray.intensity * 0.9, ray.color), None

        cos_theta2 = np.sqrt(1 - sin_theta2 * sin_theta2)
        reflection_coeff = 0.1

        reflected = Ray(
            intersection,
            ray.direction - 2 * np.dot(ray.direction, normal) * normal,
            ray.intensity * reflection_coeff,
            ray.color
        )

        refracted_direction = (n1 / n2) * ray.direction + (
                (n1 / n2) * cos_theta1 - cos_theta2
        ) * normal

        refracted = Ray(
            intersection,
            refracted_direction,
            ray.intensity * (1 - reflection_coeff),
            ray.color
        )

        return reflected, refracted

    def _reflect(self, ray: Ray, surface: Surface, intersection: np.ndarray) -> Optional[Ray]:
        surface_dir = surface.end - surface.start
        normal = np.array([-surface_dir[1], surface_dir[0]])
        normal = normal / np.linalg.norm(normal)

        if np.dot(normal, ray.direction) > 0:
            normal = -normal

        reflection = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        return Ray(intersection, reflection, ray.intensity * 0.9)

    def trace(self, origin: Tuple[float, float], angle_degrees: float, color: str = 'red'):
        angle_rad = np.radians(angle_degrees)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        ray = Ray(np.array(origin), direction, color=color)
        self.rays.append(self.trace_ray(ray))


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Ray Tracer")
        self.ray_configs = []
        self.MAX_RAYS = 3

        # Predefined colors for rays
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

        self.ray_tracer = RayTracer()
        self.ray_tracer.surfaces.append(Surface(
            np.array([2, 0]),
            np.array([2, 4]),
            refractive_index=1.5
        ))

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Surface controls
        surface_frame = ttk.LabelFrame(self.main_frame, text="Surface Properties", padding="5")
        surface_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(surface_frame, text="Refractive Index:").grid(row=0, column=0)
        self.ref_index = tk.Scale(surface_frame, from_=1.0, to=3.0, resolution=0.1,
                                  orient=tk.HORIZONTAL, command=self.update_surface)
        self.ref_index.set(1.5)
        self.ref_index.grid(row=0, column=1, sticky=(tk.W, tk.E))

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2)

        # Ray controls
        self.rays_frame = ttk.LabelFrame(self.main_frame, text="Ray Controls", padding="5")
        self.rays_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Button(self.rays_frame, text="Add New Ray", command=self.add_ray_controls).grid(row=0, column=0, pady=5)

        # Initial ray
        self.add_ray_controls()

        # Animation control
        ttk.Button(self.main_frame, text="Animate Rays", command=self.animate_rays).grid(row=3, column=0, pady=5)

    def add_ray_controls(self):

        frame = ttk.Frame(self.rays_frame)
        frame.grid(row=len(self.ray_configs) + 1, column=0, pady=5, sticky=(tk.W, tk.E))

        y_pos = tk.Scale(frame, from_=0, to=4, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_ray)
        y_pos.set(2)
        angle = tk.Scale(frame, from_=-45, to=45, resolution=1, orient=tk.HORIZONTAL, command=self.update_ray)
        angle.set(0)

        y_pos.grid(row=0, column=0, padx=5)
        angle.grid(row=0, column=1, padx=5)

        color = self.colors[len(self.ray_configs) % len(self.colors)]
        self.ray_configs.append((y_pos, angle, color))
        self.update_ray()


    def update_surface(self, *args):
        self.ray_tracer.surfaces[0].refractive_index = self.ref_index.get()
        self.update_ray()

    def update_ray(self, *args):
        self.ray_tracer.clear_rays()
        for y_pos, angle, color in self.ray_configs:
            self.ray_tracer.trace((0, y_pos.get()), angle.get(), color)
        self.update_plot()

    def animate_rays(self):
        for angle in range(-45, 46, 5):
            for _, angle_scale, _ in self.ray_configs:
                angle_scale.set(angle)
            self.root.update()
            self.root.after(50)

    def update_plot(self):
        self.ax.clear()

        # Plot surface
        surface = self.ray_tracer.surfaces[0]
        self.ax.plot([surface.start[0], surface.end[0]],
                     [surface.start[1], surface.end[1]],
                     color='lightblue', linewidth=2)

        # Plot rays
        for ray_path in self.ray_tracer.rays:
            points = np.array([ray.origin for ray in ray_path])
            last_point = ray_path[-1].origin + ray_path[-1].direction
            points = np.vstack([points, [last_point]])

            color = ray_path[0].color
            self.ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.5)
            self.ax.plot(points[0, 0], points[0, 1], 'go')

        self.ax.grid(True)
        self.ax.set_xlim(-1, 5)
        self.ax.set_ylim(-1, 5)
        self.ax.set_aspect('equal')
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()