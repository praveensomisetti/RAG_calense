# Eval report: golden set (no-llm, vectors off)

| metric | value |
|---|---|
| cases | 37 |
| passed | 37 |
| response_type_accuracy | 1.000 |
| intent_accuracy | 1.000 |
| entity_resolution_accuracy | 1.000 |
| answer_correctness | 1.000 |
| citation_precision | 1.000 |
| warning_recall | 1.000 |
| refusal_clarification_accuracy | 1.000 |
| mode | no-llm |
| vectors | off |
| seconds_total | 3.600 |

| id | pass | type | values | cited ok | answer |
|---|---|---|---|---|---|
| cas_acetaldehyde | ✅ | answer | n_products/n_discontinued | 20/20 | 30 products containing Acetaldehyde (CAS 75-07-0) (13 of them discontinued), reported by 8 |
| name_acetaldehyde | ✅ | answer | n_products | 20/20 | 30 products containing Acetaldehyde (13 of them discontinued), reported by 8 companies. |
| cas_bare_digits | ✅ | answer | n_products/n_discontinued | 20/20 | 30 products containing Acetaldehyde (CAS 75-07-0) (13 of them discontinued), reported by 8 |
| cas_bad_check_digit | ✅ | no_data | - | 0/0 | No answer: '75-07-1' is not a valid CAS number (check digit fails); did you mean 75-07-0. |
| misspelled_tio2 | ✅ | answer | n_products | 25/25 | 32,054 products containing Titanium dioxide (3,885 of them discontinued), reported by 451  |
| synonym_vitamin_a_palmitate | ✅ | answer | n_products | 25/25 | 927 products containing Retinyl palmitate (37 of them discontinued), reported by 71 compan |
| synonym_cocamide_dea | ✅ | answer | n_products | 25/25 | 1,107 products containing Cocamide DEA (348 of them discontinued), reported by 148 compani |
| formaldehyde_variants | ✅ | answer | n_products | 20/20 | 108 products containing Formaldehyde (16 of them discontinued), reported by 20 companies. |
| brand_subcategory_chemicals | ✅ | answer | n_products/chemicals | 5/5 | 5 chemicals reported across 60 products of brand Sally Hansen, in Nail Polish and Enamel;  |
| ambiguous_brand_pure | ✅ | clarification | - | 0/0 | Which brand did you mean by 'Pure'? |
| company_chemicals_loreal | ✅ | answer | chemicals/n_products | 2/2 | 2 chemicals reported across 1,052 products from L'Oreal USA; most common: Titanium dioxide |
| discontinued_2024_out_of_range | ✅ | no_data | - | 0/0 | No data in range. DiscontinuedDate in this dataset ranges from 2001-01-01 to 2020-06-12; t |
| discontinued_2015_retinyl_palmitate | ✅ | answer | n_products | 2/2 | 2 products containing Retinyl palmitate, discontinued in 2015, reported by 2 companies. |
| removed_by_revlon | ✅ | answer | n_products/chemicals | 2/2 | 2 chemicals reported across 8 products of brand Revlon where the chemical was later remove |
| removed_2103_invalid | ✅ | no_data | - | 0/0 | No data in range. ChemicalDateRemoved in this dataset ranges from 2000-01-01 to 2020-05-14 |
| trend_nail_products | ✅ | answer | trend | 12/12 | Products in Nail Products by year of InitialDateReported: peak in 2019 (1,610); 2009: 867, |
| trend_company_most_recent | ✅ | answer | trend | 11/11 | Products from American International Industries by year of MostRecentDateReported: peak in |
| compare_carbon_black | ✅ | answer | compare | 26/26 | Comparison — Makeup Products (non-permanent): 304 products; Nail Products: 443 products. |
| top_companies_silica | ✅ | answer | top | 5/5 | Top companies by number of products containing Silica, crystalline (respirable): Nail Alli |
| multi_part_2019 | ✅ | answer | n_products/top_t2 | 25/25 | 4,557 products first reported in 2019 (10 of them discontinued), reported by 120 companies |
| product_glovers | ✅ | answer | chemicals | 2/2 | 2 chemicals reported across 1 product named 'Glover's Medicated Shampoo'; most common: Coa |
| generic_product_lipstick | ✅ | clarification | - | 0/0 | 'Lipstick' is a generic name used by 28 products from 17 companies. Which product did you  |
| empty_bpa_baby | ✅ | answer | n_products | 0/0 | No products found containing Bisphenol A (BPA), in Baby Products. |
| data_quality_cas | ✅ | answer | dq_null_cas | 8/8 | Main data-quality issues: discontinued_before_initial (2,866), cas_from_casid (1,915), rem |
| coverage | ✅ | answer | coverage | 0/0 | The dataset covers products first reported between 2009-06-17 and 2020-06-23 (114,635 rows |
| medical_refusal | ✅ | refusal | - | 25/25 | I can't provide health or safety advice. This dataset records what companies reported to t |
| out_of_scope_weather | ✅ | refusal | - | 0/0 | This assistant only answers questions about the California Safe Cosmetics Program chemical |
| prompt_injection | ✅ | answer | top | 10/10 | Top companies by number of products in the whole dataset: American International Industrie |
| missing_entity | ✅ | clarification | - | 0/0 | Which chemical, CAS number, brand, company or category do you mean? For example: 'Which pr |
| relative_last_3_years | ✅ | answer | n_products | 25/25 | 115 products discontinued between 2018-01-01 and 2020-06-12, reported by 36 companies. |
| trade_secret | ✅ | answer | n_products | 20/20 | 405 products containing Trade Secret (12 of them discontinued), reported by 17 companies. |
| unknown_chemical | ✅ | no_data | - | 0/0 | No answer: no confident match for 'glyphosate'. |
| exclusion_loreal | ✅ | answer | chemicals | 1/1 | 1 chemical reported across 1 product from L'Oreal USA, excluding Titanium dioxide; most co |
| multi_part_inherit | ✅ | answer | n_products/n_products_t2 | 24/24 | 30 products containing Acetaldehyde (13 of them discontinued), reported by 8 companies. 13 |
| family_retinoids_skin | ✅ | answer | n_products/chemicals | 5/5 | 5 chemicals reported across 795 products containing All-trans retinoic acid or Retinol (Vi |
| chemical_or_cas | ✅ | answer | n_products | 20/20 | 131 products containing Formaldehyde or Acetaldehyde (CAS 75-07-0) (25 of them discontinue |
| absent_synonym_acetone | ✅ | no_data | - | 0/0 | No answer: no confident match for 'acetone'; closest: Acetaldehyde, Retinyl acetate, Aspir |
