import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# Toujours afficher tous le contenu du Dataframe
pd.set_option('display.max_columns', None)

def display_pearson_matrix(df, title, annot=False):
    # Calcul de la matrice de corrélation
    corr = df.corr(method='pearson')

    # Taille de la figure
    plt.figure(figsize=(12, 8))

    # Heatmap
    sns.heatmap(
        corr,
        annot=annot, # affiche les coefficients dans les cases
        fmt=".2f",  # deux décimales
        cmap="coolwarm",  # couleurs rouge/bleu
        center=0,  # centre la palette à 0
        cbar_kws={'label': 'Corrélation'}
    )

    plt.title(title, fontsize=16)
    plt.show()

def display_pearson_correlation(df, target_var, vars):
    print(f"\nVariables analysées: {len(vars)}")
    print(vars)

    # Calculer les corrélations
    correlations = []
    for var in vars:
        corr_coef, p_value = pearsonr(df[var], df[target_var])
        correlations.append({
            'variable': var,
            'correlation': corr_coef,
            'p_value': p_value,
            'abs_correlation': abs(corr_coef)
        })

    # Convertir en DataFrame et trier par corrélation absolue décroissante
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)

    print("\n=== CORRÉLATIONS DE PEARSON AVEC LE DÉPART DE L'ENTREPRISE ===")
    print("(Triées par corrélation absolue décroissante)")
    print("-" * 80)
    for _, row in corr_df.iterrows():
        significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row[
                                                                                                          'p_value'] < 0.05 else ""
        print(f"{row['variable']:<40} | r = {row['correlation']:>7.4f} | p = {row['p_value']:>7.4f} {significance}")

def display_pairplot(df, hue, title):
    sns.pairplot(
        df,
        hue=hue,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 25},
        palette=['blue', 'red']
    ).fig.suptitle(
        title,
        y=1.02, fontsize=14
    )

    plt.show()

def display_spearman_matrix(df, target_var):
    spearman_corr = df.drop(target_var, axis=1).corr(method='spearman')

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
    sns.heatmap(
        spearman_corr,
        mask=mask,
        annot=True,
        cmap='RdBu_r',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={"shrink": .8}
    )
    plt.title('Matrice de Corrélation de Spearman\n(Mesure des relations monotones)',
              fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 3. ANALYSE DES CORRÉLATIONS AVEC LA VARIABLE CIBLE
    target_correlations = []
    for col in df.columns:
        if col != 'a_quitte_l_entreprise':
            corr, p_value = spearmanr(df[col], df[target_var])
            target_correlations.append({
                'Variable': col,
                'Correlation_Spearman': corr,
                'P_value': p_value
            })

    corr_df = pd.DataFrame(target_correlations)
    corr_df = corr_df.sort_values('Correlation_Spearman', key=abs, ascending=False)

    plt.figure(figsize=(12, 8))
    colors = ['red' if x < 0 else 'blue' for x in corr_df['Correlation_Spearman']]
    bars = plt.barh(range(len(corr_df)), corr_df['Correlation_Spearman'], color=colors, alpha=0.7)
    plt.yticks(range(len(corr_df)), corr_df['Variable'])
    plt.title('Corrélations de Spearman avec la variable cible')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

