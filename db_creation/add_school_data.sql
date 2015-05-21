SELECT cohort.*, g6.*, g7.*, g8.*, g9.*, g10.*, g11.*, g12.*
FROM cohort1_all AS cohort
LEFT JOIN school_data AS g6 ON cohort.g6_school_id = g6.school 
LEFT JOIN school_data AS g7 ON cohort.g7_school_id = g7.school 
LEFT JOIN school_data AS g8 ON cohort.g8_school_id = g8.school
LEFT JOIN school_data AS g9 ON cohort.g9_school_id = g9.school
LEFT JOIN school_data AS g10 ON cohort.g10_school_id = g10.school
LEFT JOIN school_data AS g11 ON cohort.g11_school_id = g11.school
LEFT JOIN school_data AS g12 ON cohort.g12_school_id = g12.school);

