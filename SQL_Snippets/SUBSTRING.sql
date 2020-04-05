  select 
	SUBSTRING([date of Sale],1,2) "month", 
	SUBSTRING([date of Sale],4,2) "day", 
	SUBSTRING([date of Sale],7,4) "year", 
	substring(MSISDN,2,10) "SUBSCRIBER_NO"
into #LIGHTS_SUBBED from #LIGHTS

--- First clause is name of column,
---- Second clause is START of STRING
-----  THIRD clause is END of STRING