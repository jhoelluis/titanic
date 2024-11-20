import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import unicodedata
from functools import lru_cache

class TitanicPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.name_to_features = {}
        self.features = [
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_Q', 'Embarked_S', 'IsMother', 'IsMrs',
            'FamilySize', 'IsAlone', 'Fare_Bin', 'Age_Bin'
        ]

    @staticmethod
    @lru_cache(maxsize=1000)
    def normalize_text(text):
        """Normaliza texto con caché para mejorar rendimiento."""
        if isinstance(text, str):
            return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        return text

    def load_data(self):
        """Carga y preprocesa los datos con manejo de errores."""
        for encoding in ['utf-8', 'latin-1']:
            try:
                train_data = pd.read_csv('train.csv', encoding=encoding)
                test_data = pd.read_csv('test.csv', encoding=encoding)
                return train_data, test_data
            except UnicodeDecodeError:
                continue
        raise ValueError("No se pudieron cargar los archivos con ningún encoding")

    def preprocess_data(self, combined_data):
        """Preprocesa los datos con optimizaciones."""
        # Aplicar normalización de texto vectorizada
        text_columns = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        for col in text_columns:
            if col in combined_data.columns:
                combined_data[col] = combined_data[col].map(self.normalize_text)

        # Extracción de títulos optimizada
        combined_data['Title'] = combined_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
            'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
            'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare', 'Mlle': 'Miss',
            'Ms': 'Miss', 'Mme': 'Mrs'
        }
        combined_data['Title'].replace(title_mapping, inplace=True)

        # Imputación de valores eficiente
        for title in combined_data['Title'].unique():
            for pclass in combined_data['Pclass'].unique():
                mask = (combined_data['Title'] == title) & (combined_data['Pclass'] == pclass)
                combined_data.loc[mask & combined_data['Age'].isna(), 'Age'] = \
                    combined_data.loc[mask, 'Age'].median()

        # Imputación vectorizada para Fare
        fare_medians = combined_data.groupby('Pclass')['Fare'].transform('median')
        combined_data['Fare'].fillna(fare_medians, inplace=True)
        
        # Imputación de Embarked
        combined_data['Embarked'].fillna(combined_data['Embarked'].mode()[0], inplace=True)

        # Feature engineering optimizado
        combined_data['Sex'] = (combined_data['Sex'] == 'male').astype(int)
        combined_data = pd.get_dummies(combined_data, columns=['Embarked'], drop_first=True)
        
        combined_data['FamilySize'] = combined_data['SibSp'] + combined_data['Parch'] + 1
        combined_data['IsAlone'] = (combined_data['FamilySize'] == 1).astype(int)
        combined_data['IsMother'] = (
            (combined_data['Sex'] == 0) & 
            (combined_data['Parch'] > 0) & 
            (combined_data['Age'] > 18)
        ).astype(int)
        combined_data['IsMrs'] = (combined_data['Title'] == 'Mrs').astype(int)
        
        # Binning optimizado
        combined_data['Fare_Bin'] = pd.qcut(combined_data['Fare'], 4, labels=False)
        combined_data['Age_Bin'] = pd.qcut(combined_data['Age'], 5, labels=False)
        
        return combined_data

    def build_model(self, input_shape):
        """Construye el modelo con configuración optimizada."""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                32, 
                input_shape=(input_shape,),
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                16,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def train(self):
        """Entrena el modelo con configuración optimizada."""
        # Cargar y preprocesar datos
        train_data, test_data = self.load_data()
        combined_data = pd.concat([train_data, test_data], sort=False)
        combined_data = self.preprocess_data(combined_data)

        # Separar datos
        train_data = combined_data[combined_data['Survived'].notna()]
        test_data = combined_data[combined_data['Survived'].isna()]
        
        self.name_to_features = train_data.set_index('Name').to_dict('index')
        
        X_train = train_data[self.features].values
        y_train = train_data['Survived'].values
        X_test = test_data[self.features].values

        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Split con stratify
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_scaled, y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )

        # Construir y compilar modelo
        self.model = self.build_model(X_train.shape[1])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks optimizados
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5
            )
        ]

        # Entrenar modelo
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Guardar predicciones
        test_data['Survived'] = self.model.predict(X_test_scaled, verbose=0).round().astype(int)
        test_data[['PassengerId', 'Survived']].to_csv(
            'submission.csv',
            index=False,
            encoding='utf-8'
        )

        # Evaluar modelo
        loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print("\n=== Métricas del modelo ===")
        print(f'Precisión en validación: {accuracy:.4f}')
        print(f'Precisión final en entrenamiento: {history.history["accuracy"][-1]:.4f}')
        print(f'Precisión final en validación: {history.history["val_accuracy"][-1]:.4f}')

    def predict_survival(self, name):
        """Predice la supervivencia de un pasajero."""
        normalized_name = self.normalize_text(name)
        
        if normalized_name not in self.name_to_features:
            return "Pasajero no encontrado en los datos."
        
        passenger = self.name_to_features[normalized_name]
        features_list = [passenger.get(feat, 0) for feat in self.features]
        
        scaled_features = self.scaler.transform([features_list])
        probability = self.model.predict(scaled_features, verbose=0)[0][0]
        
        survived = "sobrevivió" if probability > 0.5 else "no sobrevivió"
        return f"El modelo predice que {name} {survived} con una probabilidad de {probability:.2f}"

# Uso del modelo
if __name__ == "__main__":
    predictor = TitanicPredictor()
    predictor.train()