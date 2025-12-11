import gymnasium as gym
import supersuit
print(f"Gymnasium version: {gym.__version__}")
print(f"Supersuit version: {supersuit.__version__}")
try:
    print(f"gym.wrappers.vector.RecordEpisodeStatistics: {gym.wrappers.vector.RecordEpisodeStatistics}")
    print(f"gym.wrappers.RecordEpisodeStatistics: {gym.wrappers.RecordEpisodeStatistics}")
    print(f"Is vector same as normal? {gym.wrappers.vector.RecordEpisodeStatistics is gym.wrappers.RecordEpisodeStatistics}")
except AttributeError as e:
    print(f"Error accessing wrappers: {e}")
