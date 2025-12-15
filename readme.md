Jiu-Jitsu Injury Clustering Analysis
Overview
This project analyzes injury patterns among Brazilian Jiu-Jitsu athletes using clustering and statistical methods. The workflow includes data cleaning, feature engineering, clustering, statistical testing, and descriptive reporting. All scripts and outputs are designed for reproducibility and transparency.

Workflow Summary

Data Preparation (Script 01)
    Raw survey data is cleaned and processed.
    Dummy variables are created for injury location, type, mechanism, environment, and severity indicators.

Clustering Preparation (Scripts 2 & 3)
    Clustering features
    Estimates of k for k-prototypes clustering

Clustering (Script 04)
    Athletes are grouped into clusters based on demographic and training features.
    A unique identifier (respondent_id) is preserved for safe merging.
    Cluster assignments and summaries are saved for further analysis.

Merging and Feature Engineering (Script 05)
    Cluster labels are merged with raw and dummy data.
    The resulting dataset (raw_dummies_clusters.csv) is used for all subsequent analyses.

Statistical Analysis (Scripts 06 & 07)
    Normality is assessed for key variables in each cluster (Shapiro-Wilk, histograms, QQ-plots).
    Non-parametric tests (Kruskal-Wallis, Dunn post-hoc) are used for numeric variables.
    Chi-square and Fisher's exact tests are used for categorical/dummy variables.
    Cluster profiles and injury prevalence are summarized in cluster_report.csv.

Reporting
    Cluster profiles are described in detail, including typical values and injury patterns.
    Severity indicators (time away, treatment, etc.) and mechanisms/environments are included.
    Comparative tables and figures are generated for publication-ready documents.

Key Files
    data/raw/dados_respondentes_raw.csv — Original survey data
    data/processed/clustering_features.csv — Features for clustering
    results/clusters.csv — Cluster assignments
    results/raw_dummies_clusters.csv — Merged dataset with dummies and clusters
    results/normality_results.csv — Normality test results
    results/kruskal_results.csv — Kruskal-Wallis test results
    results/dunn_posthoc_<variable>.csv — Dunn post-hoc results for significant variables
    results/categorical_tests.csv — Chi-square/Fisher test results for categorical variables
    results/cluster_report.csv — Cluster profiles and injury prevalence
    results/figures/ — Histograms and QQ-plots for visual analysis

How to Reproduce
    Clone the repository.
    Place raw data in the data/raw/ directory.
    Run scripts in order

Review outputs in the results/ directory.
    Use the provided tables and figures for reporting and publication.

Main Results
    Clusters are statistically distinct in age, height, body mass, and years of jiu-jitsu.
    Injury prevalence and severity vary by cluster, with older and more experienced athletes (Cluster 1) showing more severe injuries and longer time away.
    Most common injury mechanisms are impacts during training, with knees, shoulders, and hands/fingers most affected.
    Severity indicators (time away, treatment) are highest in Cluster 1.

Authors & Contact
    Project lead: Jefferson Velasco, Daniele Detanico and Romulo Sena
    Contact: 
    
    jeffvelasco.crm@gmail.com

For questions, open an issue or contact via GitHub or email.

License
This project is released under the Apache 2.0 License.
