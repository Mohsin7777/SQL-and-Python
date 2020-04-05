

-------------------------- Template used: Devices Datamart

Drop Table #temp1
Drop Table #temp2
Drop Table #temp3
drop table #result


-------------------------- Declare Date Variables

declare @index numeric(9)
DECLARE @YEAR_DATE AS NUMERIC(4,0)
DECLARE @MONTH_DATE AS NUMERIC(2,0)

set @index = (select MAX (INDEX_NO)-1 
from STAGE.dbo.STG_TIME_DIM_MONTHLY
where GETDATE() >= SOM and GETDATE() <= EOM) 

SET @YEAR_DATE = (SELECT DISTINCT YEAR_DATE 
FROM STAGE.dbo.STG_TIME_DIM_MONTHLY 
WHERE INDEX_NO = @INDEX)

SET @MONTH_DATE = (SELECT DISTINCT MONTH_DATE 
FROM STAGE.dbo.STG_TIME_DIM_MONTHLY 
WHERE INDEX_NO = @INDEX) 

-------------------------- Pull CTNs while mapping to Asset_types

select a.transaction_datetime
,a.customer_identifier
,b.asset
into #temp1
FROM [ENTERTAINMENT].[dbo].[STG_CMS_PURCHASE_TRANS_BACKUP] a  --------- (CHANGE TO CURRENT TABLE!!)
JOIN [ENTERTAINMENT].[dbo].[Datamart_REF_Distinct_Content_Types] b
on a.content_type=b.content_type
WHERE Month(a.transaction_datetime) = @MONTH_DATE 
AND YEAR(a.transaction_datetime) = @YEAR_DATE
and a.transaction_status_indicator='successful' 



-------- 2: Get ban from Subscriber Table - Use the historical process for flexibility

select a.*
,b.customer_ban
into #temp2
from #temp1 a 
join [STAGE].[dbo].[STG_SUBSCRIBER] b
on a.customer_identifier=b.SUBSCRIBER_NO
where a.transaction_datetime > b.init_activation_date 
AND (a.transaction_datetime < b.DEACTIVATION_DATE 
OR Deactivation_date IS NULL)
and (b.PRV_BAN_MOVE_DATE < a.transaction_datetime 
or b.PRV_BAN_MOVE_DATE is null)
and (b.PRV_CTN_CHG_DATE < a.transaction_datetime 
or b.PRV_CTN_CHG_DATE is null)


------------------------------------------------------------------------------------------
-- Missing CTNs were validated to be due to missing records in the SUBSCRIBER Table ----
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
-- CTNs with more than one ban are due to incidents in which CTN was recycled  ----
------------------------------------------------------------------------------------------




-------- 3: Map to IMEI_BAN to get TAC 


select 
a.asset
,a.transaction_datetime
, a.customer_identifier
, a.customer_ban 
, b.tac
,max(b.INIT_TIME) as INIT_TIME
into #temp3
from #temp2 a
join [STAGE].[dbo].[STG_IMEI_BAN] b
on a.customer_identifier=b.SUBSCRIBER_NO
AND a.customer_ban=b.ban
WHERE a.transaction_datetime >= b.INIT_TIME 
and a.transaction_datetime < UPDATE_TIME ----------<<<<<<------------ Justin's Addition
group by a.asset
,a.transaction_datetime
, a.customer_identifier
, a.customer_ban 
, b.tac 




------------ 6: Map to Hardware_DIM on TAC



select 
a.asset
,b.[DEVICE_CLASS]
,b.[DEVICE_CLASS_DESC]
,b.MANF_DESC
,b.[MODEL_ID]
,b.[MODEL_DESC]
,b.DEVICE_TYPE_DESC
,  COUNT(DISTINCT CAST(a. ban AS VARCHAR(20)) + a.SUBSCRIBER_NO) as 'Monthly_Transactions' -- CHANGE "BAN: TO CUSTOMER_BAN AND # TO SUB_NO FOR CMS)
into #result
from #temp3 a
left outer join [STAGE].[dbo].[STG_HARDWARE_DIM] b
on a.tac=b.TAC
group by a.asset
,b.[DEVICE_CLASS]
,b.[DEVICE_CLASS_DESC]
,b.MANF_DESC
,b.[MODEL_ID]
,b.[MODEL_DESC]
,b.DEVICE_TYPE_DESC

------------------------------------------------------------------- 6: INSERT INTO Mart


INSERT INTO [ENTERTAINMENT].[dbo].[Datamart_Top_Devices]

(
	
INDEX_NO
,[Asset]
,[Device_class]
,[Description]
,[OEM]
,[Model_Id]
,[Model_description]
,[DEVICE_TYPE_DESC]
,[Monthly_transactions]
)

select 
@Index  
,asset
,[DEVICE_CLASS]
,[DEVICE_CLASS_DESC]
,MANF_DESC
,[MODEL_ID]
,[MODEL_DESC]
,DEVICE_TYPE_DESC
,MONTHLY_TRANSACTIONS

from #result