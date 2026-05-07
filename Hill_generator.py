import os
import numpy as np
import scipy.ndimage
from PIL import Image
import mujoco
import mujoco.viewer

def create_terrain_file(filename="terrain.png", xml_filename="terrain.xml", tile_size=200, tiles=6):

    output_dir = os.path.join(os.path.dirname(__file__), "evorob", "world", "envs", "assets")
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    def make_octave(sigma):
        # Generate noise, filter on 3x3 tiled copy so edges wrap seamlessly
        raw = rng.uniform(0, 1, (tile_size, tile_size))
        filtered = scipy.ndimage.gaussian_filter(np.tile(raw, (3, 3)), sigma=sigma)
        tile = filtered[tile_size:2 * tile_size, tile_size:2 * tile_size]
        return (tile - tile.min()) / (tile.max() - tile.min())

    n_large  = make_octave(22)
    n_medium = make_octave(7)
    n_small  = make_octave(2)

    base_tile = 0.35 * n_large + 0.40 * n_medium + 0.25 * n_small
    terrain = np.tile(base_tile, (tiles, tiles))

    full = tile_size * tiles
    x = np.linspace(0, 1, full)
    y = np.linspace(0, 1, full)
    X, Y = np.meshgrid(x, y)

    # Flat spawn zone at world-center; radius shrinks proportionally so physical size stays constant
    spawn_radius = 0.10 / tiles
    dist = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    spawn_mask = np.clip(1.0 - dist / spawn_radius, 0.0, 1.0)
    terrain = terrain * (1.0 - spawn_mask)

    terrain = terrain - terrain.min()
    terrain = terrain / terrain.max()
    terrain[-1, -1] = 1
    terrain_normalized = (terrain * 255).astype(np.uint8)
    img = Image.fromarray(terrain_normalized, mode='L')

    png_path = os.path.join(output_dir, filename)
    img.save(png_path)

    world_half = 10 * tiles  # 10 m per tile → 40 m half-extent → 80 × 80 m world
    xml_content = f"""<mujoco model="terrain">
  <asset>
    <hfield name="terrain" file="{png_path}" nrow="{full}" ncol="{full}"
            size="{world_half} {world_half} 0.7 0.1"/>
  </asset>

  <worldbody>
    <light diffuse=".8 .8 .8" pos="0 0 10" dir="0 0 -1"/>
    <light diffuse=".3 .3 .3" pos="5 5 5" dir="-1 -1 -1"/>
    <geom name="terrain" type="hfield" hfield="terrain"
          pos="0 0 0" rgba="0.6 0.5 0.3 1"/>
  </worldbody>
</mujoco>"""

    xml_path = os.path.join(output_dir, xml_filename)
    with open(xml_path, "w") as f:
        f.write(xml_content)

    print(f"✓ PNG : {png_path}")
    print(f"✓ XML : {xml_path}")
    return png_path, xml_path


# === Main ===
if __name__ == "__main__":
    # create_terrain_file()

    model = mujoco.MjModel.from_xml_path("evorob/world/envs/assets/terrain.xml")
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)
