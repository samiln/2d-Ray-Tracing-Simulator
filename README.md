# 2D Ray Tracer

A Python-based interactive physics simulation demonstrating light ray behavior through optical surfaces. This program visualizes how light rays interact with refractive surfaces, showing both reflection and refraction effects in real-time.


## Features

- **Real-time Physics Simulation**
  - Ray reflection and refraction based on physical laws
  - Snell's law implementation for accurate light behavior
  - Multiple ray path visualization
  - Intensity tracking through multiple reflections/refractions

- **Interactive Controls**
  - Adjustable ray positions and angles
  - Configurable refractive index
  - Support for multiple colored rays (up to 4)
  - Real-time visualization updates
  - Ray path animation feature

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/2d-ray-tracer.git
cd 2d-ray-tracer
```

2. Install required dependencies:
```bash
pip install numpy matplotlib tkinter
```

## Usage

Run the main program:
```bash
python ray_tracer.py
```

### Controls:
- Use the "Add New Ray" button to add rays (maximum 4)
- Adjust ray positions using the "Y Pos" sliders
- Change ray directions using the "Angle" sliders
- Modify the refractive index to see different bending effects
- Click "Animate Rays" to see ray behavior across different angles

## Technical Details

### Physics Implementation

The program implements several optical physics concepts:

1. **Ray Tracing**: 
   - Tracks light ray paths through different media
   - Handles multiple reflections and refractions
   - Uses vector mathematics for accurate path calculation

2. **Optical Physics**:
   - Implements Snell's Law: n₁sin(θ₁) = n₂sin(θ₂)
   - Calculates reflection angles: θᵣ = θᵢ
   - Handles total internal reflection cases
   - Models intensity changes through multiple interactions

### Code Structure

- `Ray` class: Represents light rays with position, direction, and properties
- `Surface` class: Defines optical surfaces with refractive indices
- `RayTracer2D` class: Core physics engine for ray calculations
- `EnhancedRayTracerGUI` class: Handles the user interface and visualization

## Contributing

Contributions are welcome! Here are some ways you can contribute:

1. Adding new features (curved surfaces, new physics effects)
2. Improving the user interface
3. Optimizing the physics calculations
4. Adding new visualization options
5. Fixing bugs and issues

## Future Improvements

- Add support for curved surfaces
- Implement dispersion effects (rainbow creation)
- Add wave interference patterns
- Include polarization effects
- Add more complex optical elements (lenses, prisms)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on principles of geometric optics
- Built using Python's scientific computing libraries
- Inspired by educational physics simulations

## Contact

[Your Name] - [your.email@example.com]
Project Link: [https://github.com/yourusername/2d-ray-tracer](https://github.com/yourusername/2d-ray-tracer)
