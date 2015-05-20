## CREATE TABLES: COHORT 1

-- 2006: Grade 6
CREATE TABLE cohort1_2006 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	gradeexp int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	tardyr float,
	nsusp int,
	mobility int,
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	g6mapr float,
	g6msam varchar(100),
	wcode int
	);

\copy mcps.cohort1_2006 from 'MCPS_2006.csv' with DELIMITER ',' NULL 'NA' HEADER CSV;

-- 2007: Grade 7
CREATE TABLE cohort1_2007 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	gradeexp int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	tardyr float,
	nsusp int,
	mobility int,
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	g7mapr float,
	g7msam varchar(100),
	wcode int
	);

\copy mcps.cohort1_2007 from 'MCPS_2007' with DELIMITER ',' NULL 'NA' HEADER CSV;

--2008: Grade 8
CREATE TABLE cohort1_2008 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	gradeexp int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	tardyr float,
	nsusp int,
	mobility int,
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	g8mapr float,
	g8msam varchar(100),
	wcode int
	);

\copy mcps.cohort1_2008 from 'MCPS_2008' with DELIMITER ',' NULL 'NA' HEADER CSV;

--2009: Grade 9
CREATE TABLE cohort1_2009 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	gradeexp int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	nsusp int,
	retained int,
	mobility int,
	newmcps varchar(10),
	newus varchar(10),
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	wcode int,
	g8mapr float,
	g8msam varchar(100)
	);

\copy mcps.cohort1_2009 from 'MCPS_2009' with DELIMITER ',' NULL 'NA' HEADER CSV;

--2010: Grade 10
CREATE TABLE cohort1_2010 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	gradeexp int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	nsusp int,
	retained int,
	mobility int,
	newmcps varchar(10),
	newus varchar(10),
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	psatv int,
	psatm int,
	wcode int
	);

\copy mcps.cohort1_2010 from 'MCPS_2010' with DELIMITER ',' NULL 'NA' HEADER CSV;

--2011: Grade 11
CREATE TABLE cohort1_2011 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	gradeexp int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	nsusp int,
	retained varchar(255),
	mobility int,
	newmcps varchar(10),
	newus varchar(10),
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	psatv int,
	psatm int,
	wcode int
	);

\copy mcps.cohort1_2011 from 'MCPS_2011' with DELIMITER ',' NULL 'NA' HEADER CSV;

--2012: Grade 12
CREATE TABLE cohort1_2012 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	gradeexp int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	nsusp int,
	retained varchar(255),
	mobility int,
	newmcps varchar(10),
	newus varchar(10),
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	psatv int,
	psatm int,
	wcode int
	);

\copy mcps.cohort1_2012 from 'MCPS_2012' with DELIMITER ',' NULL 'NA' HEADER CSV;

#-------------------------------------------------------

## CREATE TABLES: COHORT 2

-- 2007: Grade 6
CREATE TABLE cohort2_2007 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	tardyr float,
	nsusp int,
	mobility int,
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	g6mapr float,
	g6msam varchar(100),
	wcode int
	);

\copy mcps.cohort2_2007 from 'MCPS_2007.csv' with DELIMITER ',' NULL 'NA' HEADER CSV;

-- 2008: Grade 7
CREATE TABLE cohort2_2008 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	tardyr float,
	nsusp int,
	mobility int,
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	g7mapr float,
	g7msam varchar(100),
	wcode int
	);

\copy mcps.cohort2_2008 from 'MCPS_2008.csv' with DELIMITER ',' NULL 'NA' HEADER CSV;

--2009: Grade 8
CREATE TABLE cohort2_2009 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	tardyr float,
	nsusp int,
	mobility int,
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	g8mapr float,
	g8msam varchar(100),
	wcode int
	);

\copy mcps.cohort2_2009 from 'MCPS_2009.csv' with DELIMITER ',' NULL 'NA' HEADER CSV;

--2010: Grade 9
CREATE TABLE cohort2_2010 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	nsusp int,
	retained int,
	mobility int,
	newmcps varchar(10),
	newus varchar(10),
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	wcode int,
	g8mapr float,
	g8msam varchar(100)
	);

\copy mcps.cohort2_2010 from 'MCPS_2010.csv' with DELIMITER ',' NULL 'NA' HEADER CSV;

--2011: Grade 10
CREATE TABLE cohort2_2011 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	nsusp int,
	retained int,
	mobility int,
	newmcps varchar(10),
	newus varchar(10),
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	psatv int,
	psatm int,
	wcode int
	);

\copy mcps.cohort2_2011 from 'MCPS_2011.csv' with DELIMITER ',' NULL 'NA' HEADER CSV;

--2012: Grade 11
CREATE TABLE cohort2_2012 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	nsusp int,
	retained int,
	mobility int,
	newmcps varchar(10),
	newus varchar(10),
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	psatv int,
	psatm int,
	wcode int
	);

\copy mcps.cohort2_2012 from 'MCPS_2012.csv' with DELIMITER ',' NULL 'NA' HEADER CSV;

--2013: Grade 12
CREATE TABLE cohort2_2013 (
	id int,
	school_name varchar(225),
	school_id int,
	pid int,
	year int,
	grade int,
	gender varchar(2),
	byrmm int,
	absrate float,
	nsusp int,
	retained int,
	mobility int,
	newmcps varchar(10),
	newus varchar(10),
	q1mpa float,
	q2mpa float,
	q3mpa float,
	q4mpa float,
	psatv int,
	psatm int,
	wcode int
	);

\copy mcps.cohort2_2013 from 'MCPS_2013.csv' with DELIMITER ',' NULL 'NA' HEADER CSV;


