

--------------------- Device Mapping for Game in Apps

select customer_identifier
into #temp1
FROM [ENTERTAINMENT].[dbo].[STG_CMS_PURCHASE_TRANS_BACKUP] 
WHERE content_title = 'Spider-Man: Total Mayhem HD'
and content_type IN ('APPPUSH', '56')
and Month(transaction_datetime) = 12
AND YEAR(transaction_datetime) = 2011
and transaction_status_indicator='successful' 


---- map to ban

SELECT a.*
, b.customer_ban 
INTO #temp2
FROM #temp1 a 
JOIN [STAGE].[dbo].[STG_SUBSCRIBER] b
ON a.[customer_identifier]=b.subscriber_no
WHERE LIVE_count=1



-------- 3: Map to Physical_Device to get IMEI_ESN 

Select
a.*
, b.IMEI_ESN
into #temp3 
from #temp2 a
join [STAGE].[dbo].[STG_PHYSICAL_DEVICE] b
on a.customer_identifier=b.subscriber_no
and a.customer_ban=b.ban



------------ 6: Map to Hardware_DIM while Substringing IMEI_ESN and counting transactions



select 
b.[DEVICE_CLASS]
,b.[DEVICE_CLASS_DESC]
,b.MANF_DESC
,b.[MODEL_ID]
,b.[MODEL_DESC]
,b.DEVICE_TYPE_DESC
, COUNT(DISTINCT CAST(a. customer_ban AS VARCHAR(20)) + a. customer_identifier) as 'Monthly_Transactions'
into #result
from #temp3 a
left outer join [STAGE].[dbo].[STG_HARDWARE_DIM] b
on SUBSTRING(a.IMEI_ESN,1,8)=b.TAC
group by 
b.[DEVICE_CLASS]
,b.[DEVICE_CLASS_DESC]
,b.MANF_DESC
,b.[MODEL_ID]
,b.[MODEL_DESC]
,b.DEVICE_TYPE_DESC


select * from #result