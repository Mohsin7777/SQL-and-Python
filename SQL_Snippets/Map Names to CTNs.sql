
SELECT DISTINCT SUBSCRIBER_NO
, FIRST_NAME
, LAST_BUSINESS_NAME
, [ADR_PRIMARY_LN]
, [ADR_CITY]
, [ADR_PROV_CODE]
, [ADR_POSTAL_CODE]
FROM STAGE.dbo.STG_SUBSCRIBER S
INNER JOIN STAGE.dbo.STG_BILLING_ACCOUNT B
ON S.CUSTOMER_BAN = B.BAN
    WHERE SUBSCRIBER_NO IN 
   
(
'2507321714',	 	 	 	 	 	
'4035614043', 	 	 	 	 	
'6472904973',	 	 	 	 	 	
'6139850666',	 	 	 	 	 	
'5195669209',	 	 	 	 	 	 
'6137934497',	 	 	 	 	 	
'6475025813',	 	 	 	 	 	
'2507446898',	 	 	 	 	 	 
'6479733442',	 	 	 	 	 	
'2049554611',	 	 	 	 	 	 
'2505803130',	 	 	 	 	 	 
'2262801067',	 	 	 	 	 	 
'6132463666',	 	 	 	 	 	 
'6046149980',	 	 	 	 	 	 
'6048650277',	 	 	 	 	 	 
'6473284012',	 	 	 	 	 	 
'2502177521',	 	 	 	 	 	 
'4168166232',	 	 	 	 	 	 
'5199020062',	 	 	 	 	 	 
'4162724550'	 	 	 	 	 	 
)
   AND LIVE_COUNT = 1



