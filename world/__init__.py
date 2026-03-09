"""World simulation — terrain, pheromones, food sources, and obstacles."""

from world.food import FoodSource
from world.obstacle import Obstacle, generate_convex_polygon
from world.world import World, NestRegion, TerrainType

__all__ = [
    "FoodSource",
    "Obstacle",
    "World",
    "NestRegion",
    "TerrainType",
    "generate_convex_polygon",
]
