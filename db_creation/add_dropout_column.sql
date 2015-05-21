ALTER TABLE cohort1_all
ADD COLUMN dropout int,
ADD COLUMN g6_dropout int,
ADD COLUMN g7_dropout int,
ADD COLUMN g8_dropout int,
ADD COLUMN g9_dropout int,
ADD COLUMN g10_dropout int,
ADD COLUMN g11_dropout int,
ADD COLUMN g12_dropout int;

UPDATE cohort1_all
SET g6_dropout = 1
WHERE g6_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g6_dropout = 0
WHERE g6_wcode NOT IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g7_dropout = 1
WHERE g7_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g7_dropout = 0
WHERE g7_wcode NOT IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g8_dropout = 1
WHERE g8_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g8_dropout = 0
WHERE g8_wcode NOT IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g9_dropout = 1
WHERE g9_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g9_dropout = 0
WHERE g9_wcode NOT IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g10_dropout = 1
WHERE g10_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g10_dropout = 0
WHERE g10_wcode NOT IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g11_dropout = 1
WHERE g11_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g11_dropout = 0
WHERE g11_wcode NOT IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g12_dropout = 1
WHERE g12_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET g12_dropout = 0
WHERE g12_wcode NOT IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET dropout = 1
WHERE g6_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85) or
g7_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85) or
g8_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85) or
g9_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85) or
g10_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85) or
g11_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85) or
g12_wcode IN (30,31,32,33,34,35,36,37,38,39,40,44,46,50,71,85);

UPDATE cohort1_all
SET dropout = 0
WHERE dropout IS NULL;