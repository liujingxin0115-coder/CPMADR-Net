Data provenance, preprocessing, MedDRA mapping rules, and split specification



This document describes the data sources, processing steps, and reconstruction procedure for the heterogeneous pharmacological graph used in CPMADR-Net. The graph contains 25,341 nodes and 149,224 edges, including 255 Chinese patent medicines (CPMs), 171 adverse drug reactions (ADRs), and 2,289 positive CPM–ADR pairs.



1\. Primary data sources, versions, and access dates



Drug monographs were obtained from the China Medical Information Platform (中国医药信息查询平台, https://www.dayi.org.cn/) on 2025-10-19. This source provided the screening pool of 20,259 drug records, the monographs of the 255 CPMs, their prescription compositions, and their clinical ADR text sections. The platform record ID and monograph URL (https://www.dayi.org.cn/drug/ID) of each CPM are listed in data/nodes/cpm\_nodes\_255.csv. The raw HTML pages are not included in the data package.



The Pharmacopoeia of the People's Republic of China (2020 Edition, Volume I; https://www.chp.org.cn/) was accessed on 2025-10-23. It was used to verify CPM formulations, solid herb dosages in grams, and the mass-conversion ratios of liquid formulations. The processed gram dosages (dosage\_g) and normalized dosage weights (dosage\_weight\_log\_share) are provided in data/edges/edge\_cpm\_herb\_1844.csv.



HERB 2.0 (2025 release; http://herb.ac.cn/) was accessed on 2026-01-15. It was used to standardize herbs to HERB identifiers and to obtain herb–ingredient and ingredient–target associations. The standardized entity mappings and processed edges are provided in data/nodes/ and data/edges/.



ITCM (version 1.0; http://itcm.biotcm.net/) was accessed on 2026-01-15. It provided supplementary herb–ingredient and ingredient–target associations. The processed edges are provided in data/edges/.



TCMSP (version 2.3; https://old.tcmsp-e.com/tcmsp.php) was accessed on 2026-02-02. It provided supplementary herbal chemical ingredients and their SMILES structures. The standardized SMILES strings are provided in data/nodes/ingredient\_nodes\_15040.csv.



ADReCS (http://bioinf.xmu.edu.cn/ADReCS/) was accessed on 2026-02-19. It was used as a reference for cross-checking ADR terms and their hierarchical standardization. The 171 mapped ADR nodes are provided in data/nodes/adr\_nodes\_171.csv.



MedDRA (version 28.0; https://www.meddra.org/) was accessed on 2026-02-13. It was used to map all ADR labels to 8-digit preferred term (PT) codes and their system organ classes (SOCs). Only the 171 PT and SOC labels used in this study and the study-specific mapping lexicon are provided, in data/nodes/adr\_nodes\_171.csv and data/mappings/. The full MedDRA hierarchy files are proprietary and are not redistributed.



The Open Targets Platform (https://platform.opentargets.org/) was accessed on 2026-02-15. Release 25.12 supplied the FAERS-derived target–adverse event signals used for the auxiliary target–ADR relation. See section 4 for the exact input file and reconstruction procedure.



HGNC (https://www.genenames.org/) was accessed on 2025-10-23. It was used to standardize all 9,492 protein targets to official human gene symbols. The full target list is provided in data/nodes/target\_nodes\_9492.csv.



2\. CPM inclusion and exclusion, from 20,259 records to 255 CPMs



The 255 CPMs were selected from the China Medical Information Platform through one retrieval step followed by four filtering steps.



First, 20,259 drug monographs were retrieved, covering chemical drugs, single Chinese herbal pieces, and CPMs.



Second, chemical drugs and biologics were removed according to drug category, leaving 10,360 records.



Third, the records were restricted to CPMs whose monographs contained an explicit ADR description from post-marketing data or the package insert, leaving 1,883 CPMs. Entries with an empty ADR field or with only the statement "尚不明确" (not yet clear) were excluded.



Fourth, only CPMs whose complete herb composition and quantitative dosages could be verified against the Chinese Pharmacopoeia (2020 Edition, Volume I) or the official package insert were retained, leaving 383 CPMs. Solid herbs were recorded directly in grams. Liquid or other non-mass entries, such as milliliters of herbal oil or extract, were converted to equivalent raw-herb mass in grams using the preparation ratios given in the corresponding Pharmacopoeia monograph. Within each prescription, herb dosages were normalized by log proportion (Eq. 1). 



Fifth, 128 CPMs were excluded because at least one constituent, such as an uncommon mineral or animal material or an unindexed excipient, had no standardized chemical-ingredient record in HERB 2.0, ITCM, or TCMSP. This left the final 255 CPMs. All constituent herbs of these 255 CPMs map to 383 standardized herb nodes, listed in data/nodes/cpm\_nodes\_255.csv and data/nodes/herb\_nodes\_383.csv.

3. Extraction and coding of CPM–ADR labels

We extracted adverse reaction descriptions from the China Medical Information Platform drug monographs and mapped them to MedDRA v28.0 Preferred Terms (PTs) using the mapping rules and study lexicon. Each CPM–PT pair was counted once, yielding 2,289 pairs.

Files in the GitHub repository
CPM–ADR pairs — data/edges/edge_cpm_adr_2289.csv

4. Open Targets source and auxiliary target–ADR edges

We used the FAERS-derived openfda_significant_adverse_target_reactions dataset from Open Targets Platform release 25.12.

Archived Parquet file
https://open-targets-public-data-releases.s3.amazonaws.com/platform/25.12/output/openfda_significant_adverse_target_reactions/part-00000-f8f70bae-9efa-4444-a0c9-13875f83ef31-c000.snappy.parquet
SHA-256 5a1e4eadfbaecea7a5011496c1378e4f3a97327967d9a0960d39f5cf257e3d8f

We matched English event names to MedDRA PT names after converting them to lowercase and trimming surrounding whitespace. We mapped Ensembl IDs to approved gene symbols and retained targets present in the graph. We applied no additional score threshold. Reconstruction reproduced all 4,356 released rows, representing 4,345 distinct target–PT pairs. These auxiliary signals provide context for prediction and do not establish that a CPM caused an adverse reaction through the target.

Released edge file — data/edges/edge_target_adr_4356.csv


