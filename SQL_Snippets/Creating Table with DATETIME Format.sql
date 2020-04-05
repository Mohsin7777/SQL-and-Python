
--- Step #1 ----- SUBSTRING to seperate DATES in the column

     select 
	SUBSTRING([date of Sale],1,2) "month", 
	SUBSTRING([date of Sale],4,2) "day", 
	SUBSTRING([date of Sale],7,4) "year", 
into #example2 from #example1



----------- Step #2 ------- CREATE Table with Proper Formay

CREATE TABLE entertainment.dbo.Example
(
	DATE_OF_SALE DATETIME,
	SUBSCRIBER_NO VARCHAR(25))
	
INSERT INTO entertainment.dbo.Example
(
	DATE_OF_SALE,
	SUBSCRIBER_NO )

SELECT 
	[YEAR] + '-' + [MONTH] + '-' + [DAY],
	SUBSCRIBER_NO

FROM #example2



