# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configuration de la page Streamlit
st.set_page_config(page_title="ANALYSE_DE_DONNEES", layout="wide")

# Titre de la page avec formatage HTML pour le style
st.sidebar.image("image/ml.png", caption="MULTI-VARIATION D'ANALYSE")
st.markdown("<h1 style='text-align: center; font-size: 40px;'>ANALYSE PREDICTIVE DES EVALUATIONS FISCALES 💹</h1>", unsafe_allow_html=True)

# Téléchargement de fichier (Excel ou CSV)
uploaded_file = st.file_uploader("Importez un fichier pour l'analyse", type=["xlsx", "xls", "csv"])

# Vérification de la présence d'un fichier téléchargé
if uploaded_file is not None:
    # Récupération de l'extension de fichier
    file_extension = uploaded_file.name.split('.')[-1]

    # Lecture du fichier en fonction de son extension
    if file_extension in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file)
    elif file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Format de fichier non pris en charge. Veuillez importer un fichier Excel ou CSV.")

    # Initialisation des colonnes manquantes avec des zéros
    columns_required = [
        'CA_IR', 'CA_TVA', 'CA_exportation', 'CA_local', 'DCOM_recoup', 
        'CA_achat_majoré', 'TVA_déductible_local', 'TVA_collectée_tiers',
        'TVA_import_biens', 'TVA_import_services', 'TVAI_déclarée', 'IRI_déclarée',
        'TVA_collectée_local', 'TVA_déduite_tiers', 'Régime', 'Poids_impôt_secteur',
        'Poids_impôt_contribuable', 'CA_DCOM', 'Montant_redressement_principal'
    ]
    for col in columns_required:
        if col not in df.columns:
            df[col] = 0

    # Conversion des colonnes en numérique pour éviter les erreurs de type
    colonnes_a_convertir = [
        'Exportations_recoupées', 'CA_exportation', 'CA_IR', 'CA_TVA', 'CA_local',
        'DCOM_recoup', 'CA_achat_majoré', 'TVA_déductible_local', 'TVA_collectée_tiers',
        'TVA_import_biens', 'TVA_import_services', 'TVAI_déclarée', 'IRI_déclarée',
        'TVA_collectée_local', 'TVA_déduite_tiers', 'Poids_impôt_secteur',
        'Poids_impôt_contribuable', 'CA_DCOM'
    ]
    for col in colonnes_a_convertir:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Fonctions de vérification pour déterminer le type d'entreprise
    def est_vente_de_biens(activite):
        return activite in ["AGRICULTURE", "COMMERCE", "INDUSTRIEL", "MINE", "PECHERIE ELEVAGE", "PETROLIER", "TABACS ET ALCOOL"]

    def est_prestataire_services(activite):
        return activite in ["ASSURANCE", "BANQUE", "BATIMENT TRAVAUX PUBLICS", "FINANCES", "HOTELLERIE", "IMMOBILIER", "PRESTATION DE SERVICE", "TELECOMMUNICATION", "TOURISTIQUE", "TRANSPORT"]

    # Fonctions pour le calcul des indicateurs financiers
    def calcul_ecart_ir_tva(row):
        if est_vente_de_biens(row['Activite']):
            return abs(row['CA_IR'] - row['CA_TVA'])
        elif est_prestataire_services(row['Activite']):
            return row['CA_IR'] - row['CA_TVA'] - row.get('Variances_creances_clients', 0)
        return 0

    def calcul_indicateur_ir_tva(row):
        ecart = calcul_ecart_ir_tva(row)
        return ecart / max(row['CA_IR'], row['CA_TVA']) if max(row['CA_IR'], row['CA_TVA']) != 0 else 0

    def calcul_indicateur_exportation(row):
        ecart = max(row['Exportations_recoupées'] - row['CA_exportation'], 0)
        return ecart / row['CA_IR'] if row['CA_IR'] != 0 else 1

    def calcul_indicateur_operation_locale(row):
        ecart = max(max(row['DCOM_recoup'], row['Annexes_TVA_recoupées']) - row['CA_local'], 0)
        return ecart / row['CA_IR'] if row['CA_IR'] != 0 else 1

    def calcul_ecart_dcom(row):
        ecart = max(row['DCOM_recoup'] - row['CA_DCOM'], 0)
        return ecart / row['DCOM_recoup'] if row['DCOM_recoup'] != 0 else 0

    def calcul_ecart_achat_majore(row):
        ecart = max(row['CA_achat_majoré'] - row['CA_IR'], 0)
        return ecart / row['CA_IR'] if row['CA_IR'] != 0 else 1

    def calcul_ecart_tva_deductible_local(row):
        ecart = max(row['TVA_déductible_local'] - row['TVA_collectée_tiers'], 0)
        return ecart / row['TVA_déductible_local'] if row['TVA_déductible_local'] != 0 else 0

    def calcul_ecart_tva_import_biens(row):
        ecart = max(row['TVA_import_biens'] - row['TVA_import_services'], 0)
        return ecart / row['TVA_import_biens'] if row['TVA_import_biens'] != 0 else 0

    def calcul_ecart_tva_import_services(row):
        ecart = max(row['TVA_import_services'] - row['TVAI_déclarée'], 0)
        return ecart

    def calcul_correspondance_tvai_iri(row):
        max_value = max(row['TVAI_déclarée'], 2 * row['IRI_déclarée'])
        return (row['TVAI_déclarée'] - 2 * row['IRI_déclarée']) / max_value if max_value != 0 else 0

    def calcul_ecart_tva_collectee_locale(row):
        ecart = max(row['TVA_déduite_tiers'] - row['TVA_collectée_local'], 0)
        return ecart / row['TVA_déduite_tiers'] if row['TVA_déduite_tiers'] != 0 else 0

    def calcul_permanence_credit_tva(row):
        return "Permanence du crédit de TVA pour non-structurellement créditeurs" if row['Régime'] == "DROIT COMMUN" else 0

    def calcul_poids_impot(row):
        return row['Poids_impôt_secteur'] - row['Poids_impôt_contribuable']

    def calcul_ecart_etats_financiers_dcom(row):
        max_value = max(row['CA_DCOM'], row['CA_IR'])
        return (row['CA_DCOM'] - row['CA_IR']) / max_value if max_value != 0 else 0

    def calcul_permanence_deficit_ir(row):
        return "Permanence du déficit d'IR"

    # Application des calculs sur chaque ligne du DataFrame
    df['Ecart_IR_TVA'] = df.apply(calcul_ecart_ir_tva, axis=1)
    df['Indicateur_IR_TVA'] = df.apply(calcul_indicateur_ir_tva, axis=1)
    df['Indicateur_Exportation'] = df.apply(calcul_indicateur_exportation, axis=1)
    df['Indicateur_Operation_Locale'] = df.apply(calcul_indicateur_operation_locale, axis=1)
    df['Indicateur_DCOM'] = df.apply(calcul_ecart_dcom, axis=1)
    df['Indicateur_Achat_Majore'] = df.apply(calcul_ecart_achat_majore, axis=1)
    df['Indicateur_TVA_Deductible_Local'] = df.apply(calcul_ecart_tva_deductible_local, axis=1)
    df['Indicateur_TVA_Import_Biens'] = df.apply(calcul_ecart_tva_import_biens, axis=1)
    df['Indicateur_TVA_Import_Services'] = df.apply(calcul_ecart_tva_import_services, axis=1)
    df['Indicateur_Correspondance_TVAI_IRI'] = df.apply(calcul_correspondance_tvai_iri, axis=1)
    df['Indicateur_TVA_Collectee_Locale'] = df.apply(calcul_ecart_tva_collectee_locale, axis=1)
    df['Indicateur_Permanence_Credit_TVA'] = df.apply(calcul_permanence_credit_tva, axis=1)
    df['Indicateur_Poids_Impôt'] = df.apply(calcul_poids_impot, axis=1)
    df['Indicateur_Ecart_Etats_Financiers_DCOM'] = df.apply(calcul_ecart_etats_financiers_dcom, axis=1)
    df['Indicateur_Permanence_Déficit_IR'] = df.apply(calcul_permanence_deficit_ir, axis=1)

    # Résumé des indicateurs
    st.header("Résumé des indicateurs")
    st.write("Total des IR:", df['CA_IR'].sum())
    st.write("Total des TVA:", df['CA_TVA'].sum())
    st.write("Total des DCOM:", df['DCOM_recoup'].sum())
    st.write("Total des Exportations:", df['Exportations_recoupées'].sum())
    st.write("Total des Achats Majorés:", df['CA_achat_majoré'].sum())

    # Affichage des données avec un maximum de 1000 lignes
    max_rows_to_display = 1000
    if len(df) > max_rows_to_display:
        st.warning(f"Trop de données à afficher ({len(df)} lignes). Affichage limité à {max_rows_to_display} lignes.")
        df_display = df.head(max_rows_to_display)
    else:
        df_display = df

    # Affichage du DataFrame dans Streamlit
    st.dataframe(df_display)
    
    # Lien de téléchargement des données traitées
    download_button = st.download_button(
        label="Télécharger les données traitées",
        data=df.to_csv(index=False),
        file_name='donnees_traitees.csv',
        mime='text/csv'
    )

    # Régression linéaire
    try:
        # Sélection des colonnes pour X et y
        X = df.drop(columns=['Montant_redressement_principal']).select_dtypes(include=[np.number])
        y = df['Montant_redressement_principal']

        # Supprimer les lignes avec des valeurs manquantes dans X et y
        data = pd.concat([X, y], axis=1).dropna()
        X_cleaned = data.drop(columns=['Montant_redressement_principal'])
        y_cleaned = data['Montant_redressement_principal']

        # Vérification si les données sont vides après suppression des NaN
        if X_cleaned.empty or y_cleaned.empty:
            st.error("Les données après suppression des valeurs manquantes sont vides. Impossible d'effectuer la régression.")
        else:
            # Ajustement du modèle de régression
            model = LinearRegression()
            model.fit(X_cleaned, y_cleaned)

            # Prédictions
            predictions = model.predict(X_cleaned)

            # Affichage des résultats de la régression
            st.subheader("Résultats de la Régression Linéaire")
            st.write("Coefficients du modèle :", model.coef_)
            st.write("Intercept du modèle :", model.intercept_)
            st.write("Prédictions :", predictions)

            # Affichage des résidus
            residus = y_cleaned - predictions
            st.subheader("Visualisation des Résidus")
            plt.figure(figsize=(10, 6))
            plt.scatter(predictions, residus)
            plt.axhline(0, color='red', linestyle='--')
            plt.title('Résidus de la Régression Linéaire')
            plt.xlabel('Prédictions')
            plt.ylabel('Résidus')
            st.pyplot(plt)

            # Visualisation de la régression
            plt.figure(figsize=(10, 6))
            plt.scatter(X_cleaned.iloc[:, 0], y_cleaned, color='blue', label='Données réelles')
            plt.scatter(X_cleaned.iloc[:, 0], predictions, color='orange', label='Prédictions')
            plt.plot(X_cleaned.iloc[:, 0], predictions, color='green', label='Ligne de régression', linewidth=2)
            plt.title('Régression Linéaire')
            plt.xlabel(X_cleaned.columns[0])
            plt.ylabel('Montant redressement principal')
            plt.legend()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la régression : {e}")


# Ajout d'un nouvel upload pour un fichier avec de nouvelles features
uploaded_new_file = st.file_uploader("Importez un fichier avec de nouvelles features pour prédire le Montant_redressement_principal", type=["xlsx", "xls", "csv"])

if uploaded_new_file is not None:
    # Récupération de l'extension de fichier
    new_file_extension = uploaded_new_file.name.split('.')[-1]

    # Lecture du fichier en fonction de son extension
    if new_file_extension in ['xlsx', 'xls']:
        new_df = pd.read_excel(uploaded_new_file)
    elif new_file_extension == 'csv':
        new_df = pd.read_csv(uploaded_new_file)
    else:
        st.error("Format de fichier non pris en charge. Veuillez importer un fichier Excel ou CSV.")

    # Prétraitement des nouvelles données
    for col in columns_required:
        if col not in new_df.columns:
            new_df[col] = 0  # Initialisation des colonnes manquantes

    # Conversion des colonnes en numérique, comme dans le traitement initial
    for col in colonnes_a_convertir:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0)

    # Application des mêmes calculs d'indicateurs pour le nouveau fichier
    new_df['Ecart_IR_TVA'] = new_df.apply(calcul_ecart_ir_tva, axis=1)
    new_df['Indicateur_IR_TVA'] = new_df.apply(calcul_indicateur_ir_tva, axis=1)
    new_df['Indicateur_Exportation'] = new_df.apply(calcul_indicateur_exportation, axis=1)
    new_df['Indicateur_Operation_Locale'] = new_df.apply(calcul_indicateur_operation_locale, axis=1)
    new_df['Indicateur_DCOM'] = new_df.apply(calcul_ecart_dcom, axis=1)
    new_df['Indicateur_Achat_Majore'] = new_df.apply(calcul_ecart_achat_majore, axis=1)
    new_df['Indicateur_TVA_Deductible_Local'] = new_df.apply(calcul_ecart_tva_deductible_local, axis=1)
    new_df['Indicateur_TVA_Import_Biens'] = new_df.apply(calcul_ecart_tva_import_biens, axis=1)
    new_df['Indicateur_TVA_Import_Services'] = new_df.apply(calcul_ecart_tva_import_services, axis=1)
    new_df['Indicateur_Correspondance_TVAI_IRI'] = new_df.apply(calcul_correspondance_tvai_iri, axis=1)
    new_df['Indicateur_TVA_Collectee_Locale'] = new_df.apply(calcul_ecart_tva_collectee_locale, axis=1)
    new_df['Indicateur_Permanence_Credit_TVA'] = new_df.apply(calcul_permanence_credit_tva, axis=1)
    new_df['Indicateur_Poids_Impôt'] = new_df.apply(calcul_poids_impot, axis=1)
    new_df['Indicateur_Ecart_Etats_Financiers_DCOM'] = new_df.apply(calcul_ecart_etats_financiers_dcom, axis=1)
    new_df['Indicateur_Permanence_Déficit_IR'] = new_df.apply(calcul_permanence_deficit_ir, axis=1)

    # Supposer que 'Nouvelle_Target' n'existe pas et que nous la prédisons
    X_new = new_df.drop(columns=['Montant_redressement_principal'], errors='ignore')

    # Effectuer des prédictions
    try:
        nouvelles_predictions = model.predict(X_new)
        new_df['Montant_redressement_principal'] = nouvelles_predictions
        
        # Affichage des résultats
        st.subheader("Résultats des Prédictions")
        st.dataframe(new_df)

        # Option de téléchargement du dataset avec les prédictions
        download_button_new = st.download_button(
            label="Télécharger les données avec les prédictions",
            data=new_df.to_csv(index=False),
            file_name='donnees_avec_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la prédiction : {e}")


# Thème
hide_st_style= """
<style>
#MenuPricipale {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""