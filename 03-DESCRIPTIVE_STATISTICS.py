# 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("IRIS DATASET.csv")
df.head()

# Create Groups for Sepal Length

df['sepal_length_group'] = pd.cut(
    df['sepal_length'],
    bins=3,
    labels=["Short", "Medium", "Long"]
)

df.head()

# Group Data and Calculate Statistics

grouped_stats = df.groupby(
    'sepal_length_group',
    observed=False
)['petal_length'].agg([
    'mean',
    'median',
    'min',
    'max',
    'std'
])

grouped_stats

# Create List of Petal Length Values

group_lists = df.groupby(
    'sepal_length_group',
    observed=False
)['petal_length'].apply(list)

group_lists

# Get Unique Species Names

species_list = df['species'].unique()

species_list

# Calculate Mean for Each Species

for species in species_list:

    print("\nSpecies:", species)

    species_data = df[df['species'] == species]

    print(species_data.mean(numeric_only=True))

    # Calculate Standard Deviation

for species in species_list:

    print("\nSpecies:", species)

    species_data = df[df['species'] == species]

    print(species_data.std(numeric_only=True))

    # Calculate Percentiles

for species in species_list:

    print("\nSpecies:", species)

    species_data = df[df['species'] == species]

    print(
        species_data.quantile(
            [0.25, 0.50, 0.75],
            numeric_only=True
        )
    )

    print("Mean for all species:")

df.groupby('species').mean(numeric_only=True)

print("Standard Deviation for all species:")

df.groupby('species').std(numeric_only=True)

print("Quantiles (0.25, 0.50, 0.75) for all species:")

df.groupby('species').quantile(
    [0.25, 0.50, 0.75],
    numeric_only=True
)