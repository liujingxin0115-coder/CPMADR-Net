CPMADR-Net: Dose-Aware and Path-Enhanced Heterogeneous Graph Neural Network for Chinese Patent Medicine Adverse Drug Reaction Prediction

This repository provides the processed data used in the study and the model training and evaluation code for CPMADR-Net. The data include mappings from internal IDs to entity names and external database identifiers. They also include the full list of 2,289 positive CPM–ADR pairs with their source monograph URLs, the preserved source ADR text, and the rules used to map ADR text to MedDRA preferred terms (PTs). The five-fold cross-validation splits and all pre-generated negative samples used in the experiments are also provided.

1. Dataset overview and file list

All node and edge files are stored under data/. Each file contains the internal IDs together with Chinese or English names, external database identifiers, and source information where applicable.

Nodes

The 255 Chinese patent medicines (CPMs) are listed in data/nodes/cpm_nodes_255.csv. Each CPM has an internal ID (0 to 254), its Chinese name, its record ID on the China Medical Information Platform (dayi_drug_id), and its source URL (https://www.dayi.org.cn/drug/ID). The file also gives the number of herbs, the total prescription dosage in grams, the number of ADR edges, the degree group (tail, medium, or head), and a summary of the ADR text on the platform.

The 383 herbs are listed in data/nodes/herb_nodes_383.csv. Each herb has an internal ID (0 to 382), its Chinese, English, and Latin names, and its HERB 2.0 identifier. Herb information was obtained from the China Medical Information Platform, the Chinese Pharmacopoeia, HERB 2.0, and ITCM.

The 15,040 chemical ingredients are listed in data/nodes/ingredient_nodes_15040.csv. Each ingredient has an internal ID (0 to 15039), its name, and its SMILES string standardized with RDKit. Of these ingredients, 12,123 have a valid SMILES string for ChemBERTa encoding, and the remaining 2,917 are represented by zero vectors. Ingredient information was obtained from HERB 2.0, ITCM, and TCMSP.

The 9,492 protein targets are listed in data/nodes/target_nodes_9492.csv. Each target has an internal ID (0 to 9491) and its official HGNC gene symbol. Target information was obtained from HERB 2.0, ITCM, Open Targets, and HGNC.

The 171 adverse drug reactions (ADRs) are listed in data/nodes/adr_nodes_171.csv. Each ADR has an internal ID (0 to 170), its 8-digit MedDRA PT code, and its PT name in Chinese and English. The file also gives the 8-digit code and the Chinese and English names of the system organ class (SOC) to which each PT belongs. ADR terms were obtained from the China Medical Information Platform and standardized with ADReCS and MedDRA version 28.0.

In total, the graph contains 25,341 nodes.

Edges

The 1,844 CPM–herb edges are provided in data/edges/edge_cpm_herb_1844.csv. Each edge gives the CPM and herb IDs and names, the herb dosage in grams (dosage_g), and the normalized dosage weight defined in Eq. 1 (dosage_weight_log_share). Dosages were obtained from the China Medical Information Platform and the Chinese Pharmacopoeia.

The 36,102 herb–ingredient edges are provided in data/edges/edge_herb_ingredient_36102.csv. Each edge gives the herb and ingredient IDs and names. These associations were obtained from the Chinese Pharmacopoeia, HERB 2.0, and ITCM.

The 104,633 ingredient–target edges are provided in data/edges/edge_ingredient_target_104633.csv. Each edge gives the ingredient ID and name and the target ID and HGNC symbol. These associations were obtained from HERB 2.0 and ITCM.

The 4,356 target–ADR edges are provided in data/edges/edge_target_adr_4356.csv. Each edge gives the target ID and HGNC symbol, the ADR ID, and the MedDRA PT code and names. These associations were obtained from Open Targets.

The 2,289 CPM–ADR edges, which serve as the prediction targets, are provided in data/edges/edge_cpm_adr_2289.csv. Each edge has a row ID (0 to 2288) and gives the CPM ID, name, platform record ID, and source URL. It also gives the ADR ID, the MedDRA PT code and names, the SOC name, and the degree and degree group of the CPM. These pairs were extracted from the China Medical Information Platform and standardized to MedDRA PTs.

In total, the graph contains 149,224 edges. Row counts and SHA-256 checksums of all node and edge files are recorded in data/DATASET_MANIFEST.json.

2. Data provenance, redistribution limits, and MedDRA mapping rules

The document docs/DATA_PROVENANCE_AND_MEDDRA_RULES.md describes the data sources, their URLs and access dates, and the CPM screening steps from 20,259 records to 10,360, 1,883, 383, and finally 255 CPMs. It also describes how herb dosages were converted to grams and how SMILES strings were standardized with RDKit.

The file data/mappings/meddra_pt_171_consolidated_mapping.csv contains 171 rows, one for each ADR node (IDs 0 to 170). For each MedDRA PT, it lists the Chinese clinical expressions that were merged into that PT and example CPMs in which they occur.

The file data/mappings/cn_raw_adr_to_meddra_pt_rules_and_lexicon.csv lists 217 Chinese ADR expressions from the source text and the MedDRA PT to which each expression was mapped.

The raw HTML pages of the 20,259 records retrieved from the China Medical Information Platform are not included in this repository. The full MedDRA hierarchy files are proprietary and are not redistributed either. Instead, data/nodes/cpm_nodes_255.csv and data/edges/edge_cpm_adr_2289.csv give the monograph URL of every included CPM together with the processed labels and MedDRA PT codes used in the study. The repository therefore provides the processed study data, the preserved source text, and the mapping rules needed to check each positive pair, but not the complete raw web pages.