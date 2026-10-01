# Eval report: holdout set (no-llm, vectors off)

| metric | value |
|---|---|
| cases | 14 |
| passed | 14 |
| response_type_accuracy | 1.000 |
| intent_accuracy | None |
| entity_resolution_accuracy | 1.000 |
| answer_correctness | 1.000 |
| citation_precision | 1.000 |
| warning_recall | 1.000 |
| refusal_clarification_accuracy | 1.000 |
| mode | no-llm |
| vectors | off |
| seconds_total | 1.500 |

| id | pass | type | values | cited ok | answer |
|---|---|---|---|---|---|
| h_list_acetaldehyde | ✅ | answer | n_products/n_discontinued | 20/20 | 30 products containing Acetaldehyde or Acetaldehyde (CAS 75-07-0) (13 of them discontinued |
| h_tio2_misspelled | ✅ | answer | n_products | 25/25 | 32,054 products containing Titanium dioxide (3,885 of them discontinued), reported by 451  |
| h_retinyl_or_vit_a | ✅ | answer | n_products | 25/25 | 927 products containing Retinyl palmitate (37 of them discontinued), reported by 71 compan |
| h_loreal_ingredients | ✅ | answer | chemicals/n_products | 2/2 | 2 chemicals reported across 1,052 products from L'Oreal USA; most common: Titanium dioxide |
| h_disc_2015_vit_a_palm | ✅ | answer | n_products | 2/2 | 2 products containing Retinyl palmitate, discontinued in 2015, reported by 2 companies. |
| h_nail_yearly | ✅ | answer | trend | 12/12 | Products in Nail Products by year of InitialDateReported: peak in 2019 (1,610); 2009: 867, |
| h_silica_top5 | ✅ | answer | top | 10/10 | Top companies by number of products containing Silica, crystalline (respirable): Nail Alli |
| h_talc_2030 | ✅ | no_data | - | 0/0 | No data in range. DiscontinuedDate in this dataset ranges from 2001-01-01 to 2020-06-12; t |
| h_glovers_unquoted | ✅ | answer | chemicals | 2/2 | 2 chemicals reported across 1 product named 'Glovers Medicated Shampoo', of brand Glover's |
| h_formaldehyde_danger | ✅ | refusal | - | 25/25 | I can't provide health or safety advice. This dataset records what companies reported to t |
| h_dq_summary | ✅ | answer | dq_null_cas | 8/8 | Main data-quality issues: discontinued_before_initial (2,866), cas_from_casid (1,915), rem |
| h_joke | ✅ | refusal | - | 0/0 | This assistant only answers questions about the California Safe Cosmetics Program chemical |
| h_revlon_removed | ✅ | answer | n_products/chemicals | 2/2 | 2 chemicals reported across 8 products of brand Revlon where the chemical was later remove |
| h_bad_cas | ✅ | no_data | - | 0/0 | No answer: '13463-67-8' is not a valid CAS number (check digit fails); did you mean 13463- |
