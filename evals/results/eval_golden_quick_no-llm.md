# Eval report: golden_quick set (no-llm, vectors off)

| metric | value |
|---|---|
| cases | 10 |
| passed | 10 |
| response_type_accuracy | 1.000 |
| intent_accuracy | 1.000 |
| entity_resolution_accuracy | 1.000 |
| answer_correctness | 1.000 |
| citation_precision | 1.000 |
| warning_recall | 1.000 |
| refusal_clarification_accuracy | 1.000 |
| mode | no-llm |
| vectors | off |
| seconds_total | 1.400 |

| id | pass | type | values | cited ok | answer |
|---|---|---|---|---|---|
| cas_acetaldehyde | ✅ | answer | n_products/n_discontinued | 20/20 | 30 products containing Acetaldehyde (CAS 75-07-0) (13 of them discontinued), reported by 8 |
| synonym_vitamin_a_palmitate | ✅ | answer | n_products | 25/25 | 927 products containing Retinyl palmitate (37 of them discontinued), reported by 71 compan |
| brand_subcategory_chemicals | ✅ | answer | n_products/chemicals | 5/5 | 5 chemicals reported across 60 products of brand Sally Hansen, in Nail Polish and Enamel;  |
| ambiguous_brand_pure | ✅ | clarification | - | 0/0 | Which brand did you mean by 'Pure'? |
| discontinued_2024_out_of_range | ✅ | no_data | - | 0/0 | No data in range. DiscontinuedDate in this dataset ranges from 2001-01-01 to 2020-06-12; t |
| trend_nail_products | ✅ | answer | trend | 12/12 | Products in Nail Products by year of InitialDateReported: peak in 2019 (1,610); 2009: 867, |
| compare_carbon_black | ✅ | answer | compare | 26/26 | Comparison — Makeup Products (non-permanent): 304 products; Nail Products: 443 products. |
| medical_refusal | ✅ | refusal | - | 25/25 | I can't provide health or safety advice. This dataset records what companies reported to t |
| multi_part_inherit | ✅ | answer | n_products/n_products_t2 | 24/24 | 30 products containing Acetaldehyde (13 of them discontinued), reported by 8 companies. 13 |
| chemical_or_cas | ✅ | answer | n_products | 20/20 | 131 products containing Formaldehyde or Acetaldehyde (CAS 75-07-0) (25 of them discontinue |
