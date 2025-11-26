import pandas as pd
import numpy as np
import os
import folium
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import glob
import io
import base64
from joblib import Parallel, delayed
import re
from tqdm import tqdm
import argparse


class CreateMap: