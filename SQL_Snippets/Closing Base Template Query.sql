
---------------------- CLOSING BASE ----------------------
---------------------- TEMPLATE QUERY ------------------------



--- * #soc would contain a list of specific socs derived from the feature or switchcode 





-- FOR DATES USE THE FIRST OF THE FOLLOWING MONTH


select sa.SOC
,sa.SERVICE_TYPE
,s.soc_description
,COUNT(distinct(cast(ban as varchar(20))+subscriber_no)) as May
from STAGE.dbo.STG_SERVICE_AGREEMENT sa
inner join #soc s
	on sa.SOC = s.SOC
where EFFECTIVE_DATE <  '06/01/2011' ----------------------------------------- OR 'YEAR-MM-DD'
AND (EXPIRATION_DATE >= '06/01/2011' or EXPIRATION_DATE is null)
group by sa.SOC,sa.SERVICE_TYPE,s.soc_description

-----
