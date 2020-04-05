



-------------------- Music Subscription by RIM Devices --------------------

--- Get Current Base (subscriber count) for RIM Devices only, for these socs:

MS2GOWBR
MSPCR
OPTSRA0WS
OPTSRA0WT
OPTSRA0WU

--------------------------------------------------------------------------------



---- Step 1 >>>>TEMPA: get all the people on these SOCs (pull CTN-BAN, Soc) WHERE exp is null or > than last friday >>>>TEMPA!!! ----
---- Step 2 >>>>TEMPB: left outer join temp (!!on sub AND ban!!) with subsciber table with S.DEACTIVATION_DATE >= '11/11/2011' OR S.DEACTIVATION_DATE IS NULL)and S.SUB_STATUS IN ('A','S')
---- Step 3 >>>> Then do the following two joins to get device 
---------------->>> INNER JOIN STAGE.dbo.STG_PHYSICAL_DEVICE I ON S.SUBSCRIBER_NO=I.SUBSCRIBER_NO AND S.CUSTOMER_BAN = I.BAN
---------------->>>	INNER JOIN STAGE.dbo.STG_HARDWARE_DIM D ON I.IMEI_ITEMID = D.TAC
--------------->>> SELECT "DEVICE_CLASS_DESC, MANF_DESC,DEVICE_TYPE_DESC, MODEL_ID, MODEL_DESC" from hardware_dim
------Step 4 >>> Do a "COUNT(DISTINCT CAST(BAN AS VARCHAR(20))+SUBSCRIBER_NO)" at the end to get counted results
--------------- where device_type_desc like ‘%BLACKEBERRY%’ or rim etc  (or for all)


------?????  TEMPc: Map (tempb) BAN-CTN combination to PHYS-DEV
---- ?????? >>>>TEMPd: Take first 8 IMEI_ESN #s
---- ?????? >>>>TEMPe: Map tempC to HARDWARE DIM on TAC for device names 



Step #1

select BAN,SUBSCRIBER_NO, soc into #TEMPA
from [STAGE].[dbo].[STG_SERVICE_AGREEMENT]
WHERE (EXPIRATION_DATE IS NULL OR EXPIRATION_DATE > '11/11/2011')
AND SOC IN ('MS2GOWBR','MSPCR','OPTSRA0WS','OPTSRA0WT','OPTSRA0WU')

select * from #TEMPA


------------------ Step #2 -- Double check customer status ------------

select a.ban, a.SUBSCRIBER_NO, a.soc into #tempB
from #tempA a 
left join [STAGE].[dbo].[STG_SUBSCRIBER] b
ON a.ban = b.[CUSTOMER_BAN] and a.SUBSCRIBER_NO = b.SUBSCRIBER_NO
WHERE (b.DEACTIVATION_DATE >= '11/11/2011' OR b.DEACTIVATION_DATE IS NULL) 
and b.SUB_STATUS IN ('A','S')

select * from #tempB



------------ Step #3


select a.*, c.DEVICE_CLASS_DESC, c.MANF_DESC, c.DEVICE_TYPE_DESC, c.MODEL_ID, c.MODEL_DESC into #tempC
from #tempb a
INNER JOIN STAGE.dbo.STG_PHYSICAL_DEVICE b 
ON a.SUBSCRIBER_NO=b.SUBSCRIBER_NO AND a.BAN = b.BAN
INNER JOIN STAGE.dbo.STG_HARDWARE_DIM c ON b.IMEI_ITEMID = c.TAC


select * from #tempC



-------- Step #4 


select Device_Type_Desc
, Model_ID
, Model_Desc
, COUNT(DISTINCT CAST(BAN AS VARCHAR(20))+SUBSCRIBER_NO) as 'Current Closing Base'
from #tempC
WHERE Manf_desc = 'RIM'
group by 
 Device_Type_Desc
, Model_ID
, Model_Desc
order by 'Current Closing Base' desc
