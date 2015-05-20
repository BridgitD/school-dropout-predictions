#----------------------------------------------
# Cohort 1
#----------------------------------------------
CREATE TABLE temp6_7 AS
SELECT 
	g6.school_name AS g6_school_name,
	g6.school_id AS g6_school_id,
	g6.pid AS g6_pid,
	g6.year AS g6_year,
	g6.gradeexp AS g6_gradeexp,
	g6.grade AS g6_grade,
	g6.gender AS g6_gender,
	g6.byrmm AS g6_byrmm,
	g6.absrate AS g6_absrate,
	g6.tardyr AS g6_tardyr,
	g6.nsusp AS g6_nsusp,
	g6.mobility AS g6_mobility,
	g6.q1mpa AS g6_q1mpa,
	g6.q2mpa AS g6_q2mpa,
	g6.q3mpa AS g6_q3mpa,
	g6.q4mpa AS g6_q4mpa,
	g6.g6mapr AS g6_g6mapr,
	g6.g6msam AS g6_g6msam,
	g6.wcode AS g6_wcode,
	g7.school_name AS g7_school_name,
	g7.school_id AS g7_school_id,
	g7.pid AS g7_pid,
	g7.year AS g7_year,
	g7.gradeexp AS g7_gradeexp,
	g7.grade AS g7_grade,
	g7.gender AS g7_gender,
	g7.byrmm AS g7_byrmm,
	g7.absrate AS g7_absrate,
	g7.tardyr AS g7_tardyr,
	g7.nsusp AS g7_nsusp,
	g7.mobility AS g7_mobility,
	g7.q1mpa AS g7_q1mpa,
	g7.q2mpa AS g7_q2mpa,
	g7.q3mpa AS g7_q3mpa,
	g7.q4mpa AS g7_q4mpa,
	g7.g7mapr AS g7_g7mapr,
	g7.g7msam AS g7_g7msam,
	g7.wcode AS g7_wcode
FROM cohort1_2006 AS g6
FULL OUTER JOIN cohort1_2007 AS g7 ON g6.pid = g7.pid;

CREATE TABLE temp_middle AS
SELECT
	g67.*,
	g8.school_name AS g8_school_name,
	g8.school_id AS g8_school_id,
	g8.pid AS g8_pid,
	g8.year AS g8_year,
	g8.gradeexp AS g8_gradeexp,
	g8.grade AS g8_grade,
	g8.gender AS g8_gender,
	g8.byrmm AS g8_byrmm,
	g8.absrate AS g8_absrate,
	g8.tardyr AS g8_tardyr,
	g8.nsusp AS g8_nsusp,
	g8.mobility AS g8_mobility,
	g8.q1mpa AS g8_q1mpa,
	g8.q2mpa AS g8_q2mpa,
	g8.q3mpa AS g8_q3mpa,
	g8.q4mpa AS g8_q4mpa,
	g8.g8mapr AS g8_g8mapr,
	g8.g8msam AS g8_g8msam,
	g8.wcode AS g8_wcode
FROM temp6_7 AS g67
FULL OUTER JOIN cohort1_2008 AS g8 ON g67.g7_pid = g8.pid;

CREATE TABLE temp_9 AS
SELECT
	temp_middle.*,
	g9.school_name AS g9_school_name,
	g9.school_id AS g9_school_id,
	g9.pid AS g9_pid,
	g9.year AS g9_year,
	g9.gradeexp AS g9_gradeexp,
	g9.grade AS g9_grade,
	g9.gender AS g9_gender,
	g9.byrmm AS g9_byrmm,
	g9.absrate AS g9_absrate,
	g9.nsusp AS g9_nsusp,
	g9.retained AS g9_retained,
	g9.mobility AS g9_mobility,
	g9.newmcps AS g9_newmcps,
	g9.newus AS g9_newus,
	g9.q1mpa AS g9_q1mpa,
	g9.q2mpa AS g9_q2mpa,
	g9.q3mpa AS g9_q3mpa,
	g9.q4mpa AS g9_q4mpa,
	g9.wcode AS g9_wcode,
	g9.g8mapr AS g9_g8mapr,
	g9.g8msam AS g9_g8msam
FROM temp_middle
FULL OUTER JOIN cohort1_2009 AS g9 ON temp_middle.g8_pid = g9.pid;

CREATE TABLE temp_10 AS 
SELECT 
	temp_9.*,
	g10.school_name AS g10_school_name,
	g10.school_id AS g10_school_id,
	g10.pid AS g10_pid,
	g10.year AS g10_year,
	g10.gradeexp AS g10_gradeexp,
	g10.grade AS g10_grade,
	g10.gender AS g10_gender,
	g10.byrmm AS g10_byrmm,
	g10.absrate AS g10_absrate,
	g10.nsusp AS g10_nsusp,
	g10.retained AS g10_retained,
	g10.mobility AS g10_mobility,
	g10.newmcps AS g10_newmcps,
	g10.newus AS g10_newus,
	g10.q1mpa AS g10_q1mpa,
	g10.q2mpa AS g10_q2mpa,
	g10.q3mpa AS g10_q3mpa,
	g10.q4mpa AS g10_q4mpa,
	g10.psatv AS g10_psatv,
	g10.psatm AS g10_psatm,
	g10.wcode AS g10_wcode
FROM temp_9
FULL OUTER JOIN cohort1_2010 AS g10 ON temp_9.g9_pid = g10.pid;

CREATE TABLE temp_11 AS 
SELECT 
	temp_10.*,
	g11.school_name AS g11_school_name,
	g11.school_id AS g11_school_id,
	g11.pid AS g11_pid,
	g11.year AS g11_year,
	g11.gradeexp AS g11_gradeexp,
	g11.grade AS g11_grade,
	g11.gender AS g11_gender,
	g11.byrmm AS g11_byrmm,
	g11.absrate AS g11_absrate,
	g11.nsusp AS g11_nsusp,
	g11.retained AS g11_retained,
	g11.mobility AS g11_mobility,
	g11.newmcps AS g11_newmcps,
	g11.newus AS g11_newus,
	g11.q1mpa AS g11_q1mpa,
	g11.q2mpa AS g11_q2mpa,
	g11.q3mpa AS g11_q3mpa,
	g11.q4mpa AS g11_q4mpa,
	g11.psatv AS g11_psatv,
	g11.psatm AS g11_psatm,
	g11.wcode AS g11_wcode
FROM temp_10
FULL OUTER JOIN cohort1_2011 AS g11 ON temp_10.g10_pid = g11.pid;

CREATE TABLE cohort1_all AS 
SELECT 
	temp_11.*,
	g12.school_name AS g12_school_name,
	g12.school_id AS g12_school_id,
	g12.pid AS g12_pid,
	g12.year AS g12_year,
	g12.gradeexp AS g12_gradeexp,
	g12.grade AS g12_grade,
	g12.gender AS g12_gender,
	g12.byrmm AS g12_byrmm,
	g12.absrate AS g12_absrate,
	g12.nsusp AS g12_nsusp,
	g12.retained AS g12_retained,
	g12.mobility AS g12_mobility,
	g12.newmcps AS g12_newmcps,
	g12.newus AS g12_newus,
	g12.q1mpa AS g12_q1mpa,
	g12.q2mpa AS g12_q2mpa,
	g12.q3mpa AS g12_q3mpa,
	g12.q4mpa AS g12_q4mpa,
	g12.psatv AS g12_psatv,
	g12.psatm AS g12_psatm,
	g12.wcode AS g12_wcode
FROM temp_11
FULL OUTER JOIN cohort1_2012 AS g12 ON temp_11.g11_pid = g12.pid;

ALTER TABLE cohort1_all DROP COLUMN IF EXISTS id;
ALTER TABLE cohort1_all ADD COLUMN id serial;
ALTER SEQUENCE serial RESTART 1;
UPDATE cohort1_all SET id=nextval('serial');

#----------------------------------------------
# Cohort 2
#----------------------------------------------
CREATE TABLE temp6_7 AS
SELECT 
	g6.school_name AS g6_school_name,
	g6.school_id AS g6_school_id,
	g6.pid AS g6_pid,
	g6.year AS g6_year,
	g6.grade AS g6_grade,
	g6.gender AS g6_gender,
	g6.byrmm AS g6_byrmm,
	g6.absrate AS g6_absrate,
	g6.tardyr AS g6_tardyr,
	g6.nsusp AS g6_nsusp,
	g6.mobility AS g6_mobility,
	g6.q1mpa AS g6_q1mpa,
	g6.q2mpa AS g6_q2mpa,
	g6.q3mpa AS g6_q3mpa,
	g6.q4mpa AS g6_q4mpa,
	g6.g6mapr AS g6_g6mapr,
	g6.g6msam AS g6_g6msam,
	g6.wcode AS g6_wcode,
	g7.school_name AS g7_school_name,
	g7.school_id AS g7_school_id,
	g7.pid AS g7_pid,
	g7.year AS g7_year,
	g7.grade AS g7_grade,
	g7.gender AS g7_gender,
	g7.byrmm AS g7_byrmm,
	g7.absrate AS g7_absrate,
	g7.tardyr AS g7_tardyr,
	g7.nsusp AS g7_nsusp,
	g7.mobility AS g7_mobility,
	g7.q1mpa AS g7_q1mpa,
	g7.q2mpa AS g7_q2mpa,
	g7.q3mpa AS g7_q3mpa,
	g7.q4mpa AS g7_q4mpa,
	g7.g7mapr AS g7_g7mapr,
	g7.g7msam AS g7_g7msam,
	g7.wcode AS g7_wcode
FROM cohort2_2007 AS g6
FULL OUTER JOIN cohort2_2008 AS g7 ON g6.pid = g7.pid;

CREATE TABLE temp_middle AS
SELECT
	g67.*,
	g8.school_name AS g8_school_name,
	g8.school_id AS g8_school_id,
	g8.pid AS g8_pid,
	g8.year AS g8_year,
	g8.grade AS g8_grade,
	g8.gender AS g8_gender,
	g8.byrmm AS g8_byrmm,
	g8.absrate AS g8_absrate,
	g8.tardyr AS g8_tardyr,
	g8.nsusp AS g8_nsusp,
	g8.mobility AS g8_mobility,
	g8.q1mpa AS g8_q1mpa,
	g8.q2mpa AS g8_q2mpa,
	g8.q3mpa AS g8_q3mpa,
	g8.q4mpa AS g8_q4mpa,
	g8.g8mapr AS g8_g8mapr,
	g8.g8msam AS g8_g8msam,
	g8.wcode AS g8_wcode
FROM temp6_7 AS g67
FULL OUTER JOIN cohort2_2009 AS g8 ON g67.g7_pid = g8.pid;

CREATE TABLE temp_9 AS
SELECT
	temp_middle.*,
	g9.school_name AS g9_school_name,
	g9.school_id AS g9_school_id,
	g9.pid AS g9_pid,
	g9.year AS g9_year,
	g9.grade AS g9_grade,
	g9.gender AS g9_gender,
	g9.byrmm AS g9_byrmm,
	g9.absrate AS g9_absrate,
	g9.nsusp AS g9_nsusp,
	g9.retained AS g9_retained,
	g9.mobility AS g9_mobility,
	g9.newmcps AS g9_newmcps,
	g9.newus AS g9_newus,
	g9.q1mpa AS g9_q1mpa,
	g9.q2mpa AS g9_q2mpa,
	g9.q3mpa AS g9_q3mpa,
	g9.q4mpa AS g9_q4mpa,
	g9.wcode AS g9_wcode,
	g9.g8mapr AS g9_g8mapr,
	g9.g8msam AS g9_g8msam
FROM temp_middle
FULL OUTER JOIN cohort2_2010 AS g9 ON temp_middle.g8_pid = g9.pid;

CREATE TABLE temp_10 AS 
SELECT 
	temp_9.*,
	g10.school_name AS g10_school_name,
	g10.school_id AS g10_school_id,
	g10.pid AS g10_pid,
	g10.year AS g10_year,
	g10.grade AS g10_grade,
	g10.gender AS g10_gender,
	g10.byrmm AS g10_byrmm,
	g10.absrate AS g10_absrate,
	g10.nsusp AS g10_nsusp,
	g10.retained AS g10_retained,
	g10.mobility AS g10_mobility,
	g10.newmcps AS g10_newmcps,
	g10.newus AS g10_newus,
	g10.q1mpa AS g10_q1mpa,
	g10.q2mpa AS g10_q2mpa,
	g10.q3mpa AS g10_q3mpa,
	g10.q4mpa AS g10_q4mpa,
	g10.psatv AS g10_psatv,
	g10.psatm AS g10_psatm,
	g10.wcode AS g10_wcode
FROM temp_9
FULL OUTER JOIN cohort2_2011 AS g10 ON temp_9.g9_pid = g10.pid;

CREATE TABLE temp_11 AS 
SELECT 
	temp_10.*,
	g11.school_name AS g11_school_name,
	g11.school_id AS g11_school_id,
	g11.pid AS g11_pid,
	g11.year AS g11_year,
	g11.grade AS g11_grade,
	g11.gender AS g11_gender,
	g11.byrmm AS g11_byrmm,
	g11.absrate AS g11_absrate,
	g11.nsusp AS g11_nsusp,
	g11.retained AS g11_retained,
	g11.mobility AS g11_mobility,
	g11.newmcps AS g11_newmcps,
	g11.newus AS g11_newus,
	g11.q1mpa AS g11_q1mpa,
	g11.q2mpa AS g11_q2mpa,
	g11.q3mpa AS g11_q3mpa,
	g11.q4mpa AS g11_q4mpa,
	g11.psatv AS g11_psatv,
	g11.psatm AS g11_psatm,
	g11.wcode AS g11_wcode
FROM temp_10
FULL OUTER JOIN cohort2_2012 AS g11 ON temp_10.g10_pid = g11.pid;

CREATE TABLE cohort2_all AS 
SELECT 
	temp_11.*,
	g12.school_name AS g12_school_name,
	g12.school_id AS g12_school_id,
	g12.pid AS g12_pid,
	g12.year AS g12_year,
	g12.grade AS g12_grade,
	g12.gender AS g12_gender,
	g12.byrmm AS g12_byrmm,
	g12.absrate AS g12_absrate,
	g12.nsusp AS g12_nsusp,
	g12.retained AS g12_retained,
	g12.mobility AS g12_mobility,
	g12.newmcps AS g12_newmcps,
	g12.newus AS g12_newus,
	g12.q1mpa AS g12_q1mpa,
	g12.q2mpa AS g12_q2mpa,
	g12.q3mpa AS g12_q3mpa,
	g12.q4mpa AS g12_q4mpa,
	g12.psatv AS g12_psatv,
	g12.psatm AS g12_psatm,
	g12.wcode AS g12_wcode
FROM temp_11
FULL OUTER JOIN cohort2_2013 AS g12 ON temp_11.g11_pid = g12.pid;

DROP TABLE temp6_7;
DROP TABLE temp_middle;
DROP TABLE temp_9;
DROP TABLE temp_10;
DROP TABLE temp_11;

ALTER TABLE cohort2_all DROP COLUMN IF EXISTS id;
ALTER TABLE cohort2_all ADD COLUMN id serial;
ALTER SEQUENCE serial RESTART 1;
UPDATE cohort2_all SET id=nextval('serial');
