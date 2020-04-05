
-----------Eligibility Relationships -----------------------------------------
----- The Eligibility_relationship table gives eligibility **AND*** relationships!!!
----- The eligibiles have to be FILTERED OUT to get ONLY the RELATIONS ---
----- JOIN THE DEPENDENTS back to the SOC table to see if their type is 'D' (dependent)-----


--- Check dependents


SELECT *
FROM STAGE.dbo.STG_ELIGIBILITY_RELATION E
INNER JOIN STAGE.dbo.STG_SOC S
ON E.DEST_SOC = S.SOC
WHERE S.EXPIRATION_DATE IS NULL
AND S.SERVICE_TYPE = 'D'
AND E.SRC_SOC = 'VPIPUNLR'

SELECT A.SOC, E.DEST_SOC
FROM #CD_STYPE A
INNER JOIN STAGE.dbo.STG_ELIGIBILITY_RELATION E
ON A.SOC=E.SRC_SOC
INNER JOIN STAGE.dbo.STG_SOC S
ON E.DEST_SOC = S.SOC
WHERE S.EXPIRATION_DATE IS NULL
AND S.SERVICE_TYPE = 'D'
AND A.SOC = 'R'


------------ Detailed Example:


------------------------- Socs with Ringback Feature Code

select * into #RF
from [STAGE].[dbo].[STG_RATED_FEATURE] 			
WHERE EXPIRATION_DATE IS NULL
AND FEATURE_CODE = 'RINGBK'	

------------------------ Get Soc Details

select a.SOC, a.SOC_DESCRIPTION, a.SERVICE_TYPE
into #soc
from STAGE.dbo.STG_SOC a
inner join #RF b
	on a.SOC = b.soc
where a.EXPIRATION_DATE is null


----------------------- Map parents of dependents

select a.SRC_SOC
into #oldvaluepacksPLUSTag
from STAGE.dbo.STG_ELIGIBILITY_RELATION a
inner join #soc b
	on a.DEST_SOC = b.SOC
where b.SERVICE_TYPE = 'D'

select * from 
#oldvaluepacksPLUSTag











