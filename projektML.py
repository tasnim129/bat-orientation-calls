import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#label laden
df = pd.read_csv("labels.csv", sep=";")
df["id"] = df["Filename"].str.split(".").str[0]
df = df.drop_duplicates(subset="id")
print(df["Species"].value_counts())
#seltene Arten entfernen 
df = df[~df["Species"].isin(["Mausohr", "Wasserfledermaus"])]
print(len(df), "Bilder,", df["Species"].nunique(), "Arten")

X = []
y = []
print("Beispiel-ID:", df["id"].iloc[0])
print("Gesuchter Pfad:", os.path.join("Bilder", df["id"].iloc[0] + ".png"))
print("Was wirklich im Ordner liegt:", os.listdir("Bilder")[:5])
for i, zeile in df.iterrows():
    pfad = os.path.join("Bilder", zeile["id"] + ".png")
    if os.path.exists(pfad):
        bild = Image.open(pfad).convert("L")     # Graustufen
        bild = bild.resize((128, 64))            # gleiche Größe
        zahlen = np.array(bild) / 255.0          # Werte 0 bis 1
        X.append(zahlen)
        y.append(zeile["Species"])

X = np.array(X)       # Form: (Anzahl Bilder, 64, 128)
y = np.array(y)
print(X.shape)


plt.imshow(X[5], cmap="magma", aspect="auto")
plt.title(y[0])
plt.colorbar()
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print("Training:", len(y_train), " Test:", len(y_test))

X_train_flach = X_train.reshape(len(X_train), -1)
X_test_flach  = X_test.reshape(len(X_test), -1)

print("Vorher:", X_train.shape)
print("Nachher:", X_train_flach.shape)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train_flach, y_train)
vorhersage = baseline.predict(X_test_flach)
acc_baseline = accuracy_score(y_test, vorhersage)

print("Baseline Accuracy:", round(acc_baseline, 3))

#Decision Tree
baum = DecisionTreeClassifier(max_depth=5, random_state=42)
baum.fit(X_train_flach, y_train)
acc_baum = accuracy_score(y_test, baum.predict(X_test_flach))

print("Decision Tree Accuracy:", round(acc_baum, 3))

#Overfitting sichtbar machen
tiefen = [1, 2, 3, 5, 8, 12, 20]
train_fehler = []
test_fehler = []

for t in tiefen:
    b = DecisionTreeClassifier(max_depth=t, random_state=42)
    b.fit(X_train_flach, y_train)
    train_fehler.append(1 - b.score(X_train_flach, y_train))
    test_fehler.append(1 - b.score(X_test_flach, y_test))
    print("Tiefe", t, " Trainingsfehler:", round(train_fehler[-1], 3),
          " Testfehler:", round(test_fehler[-1], 3))
plt.figure()
plt.plot(tiefen, train_fehler, marker="o", label="Trainingsfehler")
plt.plot(tiefen, test_fehler, marker="o", label="Testfehler")
plt.xlabel("Baumtiefe (Komplexität)")
plt.ylabel("Fehlerrate")
plt.title("Decision Tree: Trainings- vs. Testfehler")
plt.legend()
#plt.show()

#nehmen wir tief 5 weil es die einfachste ist sonst gehen wir overfitting

#Random Forest
wald = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
wald.fit(X_train_flach, y_train)
acc_wald = accuracy_score(y_test, wald.predict(X_test_flach))

print("Random Forest Accuracy:", round(acc_wald, 3))
print("Trainings-Accuracy:", round(wald.score(X_train_flach, y_train), 3))

# Teil 13: FFNN
ffnn = MLPClassifier(hidden_layer_sizes=(256, 64),
                     early_stopping=True,
                     max_iter=300,
                     random_state=42)
ffnn.fit(X_train_flach, y_train)
acc_ffnn = accuracy_score(y_test, ffnn.predict(X_test_flach))

print("FFNN Accuracy:", round(acc_ffnn, 3))
print("Trainings-Accuracy:", round(ffnn.score(X_train_flach, y_train), 3))

plt.figure()
plt.plot(ffnn.loss_curve_)
plt.title("FFNN: Fehler (Loss) pro Epoche")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.show()