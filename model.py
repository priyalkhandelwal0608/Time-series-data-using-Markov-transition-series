# ============================
# Task 1: Import Required Libraries
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField
from sklearn.preprocessing import KBinsDiscretizer

# ============================
# Task 2: Import the Time Series Dataset
# ============================
# For demo purposes, let's simulate Electricity Transformer Temperature (ETT) data
# Replace this with: pd.read_csv("ETT.csv") if you have the dataset
np.random.seed(42)
time = np.arange(0, 500)
temperature = 20 + 5 * np.sin(time / 50) + np.random.normal(0, 0.5, size=500)
df = pd.DataFrame({"time": time, "temperature": temperature})

# ============================
# Task 3: Truncate and Plot the DataFrame
# ============================
df_trunc = df.iloc[:200]  # truncate for visualization
plt.figure(figsize=(10, 4))
plt.plot(df_trunc["time"], df_trunc["temperature"], label="ETT")
plt.title("Electricity Transformer Temperature (Truncated)")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()

# ============================
# Task 4: Discretize the Data
# ============================
# Discretize into bins for Markov transitions
n_bins = 5
discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
X_disc = discretizer.fit_transform(df_trunc[["temperature"]]).astype(int).flatten()

# ============================
# Task 5: Create the Adjacency Matrix
# ============================
adj_matrix = np.zeros((n_bins, n_bins))
for (i, j) in zip(X_disc[:-1], X_disc[1:]):
    adj_matrix[i, j] += 1

plt.imshow(adj_matrix, cmap="Blues")
plt.title("Adjacency Matrix")
plt.colorbar()
plt.show()

# ============================
# Task 6: Calculate the Markov Matrix
# ============================
markov_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
markov_matrix = np.nan_to_num(markov_matrix)  # handle division by zero

plt.imshow(markov_matrix, cmap="viridis")
plt.title("Markov Transition Matrix")
plt.colorbar()
plt.show()

# ============================
# Task 7: Create the Markov Transition Field
# ============================
mtf = MarkovTransitionField(n_bins=n_bins)
X_mtf = mtf.fit_transform(df_trunc[["temperature"]].values)

plt.imshow(X_mtf[0], cmap="plasma")
plt.title("Markov Transition Field")
plt.colorbar()
plt.show()

# ============================
# Task 8: Visualize the Markov Transition Field
# ============================
plt.figure(figsize=(6, 6))
plt.imshow(X_mtf[0], cmap="hot", interpolation="nearest")
plt.title("MTF Visualization")
plt.colorbar()
plt.show()

# ============================
# Task 9: Downsample the Markov Transition Field
# ============================
from skimage.transform import resize
X_mtf_downsampled = resize(X_mtf[0], (50, 50), anti_aliasing=True)

plt.imshow(X_mtf_downsampled, cmap="inferno")
plt.title("Downsampled MTF")
plt.colorbar()
plt.show()

# ============================
# Task 10: Plot Self-Transition Probabilities
# ============================
self_transitions = np.diag(markov_matrix)
plt.plot(range(n_bins), self_transitions, marker="o")
plt.title("Self-Transition Probabilities")
plt.xlabel("State Bin")
plt.ylabel("Probability")
plt.show()

# ============================
# Task 11: Extract Insights
# ============================
print("Insights:")
print(f"- Number of bins: {n_bins}")
print(f"- Strongest self-transition at bin {np.argmax(self_transitions)}")
print(f"- Average self-transition probability: {np.mean(self_transitions):.3f}")
