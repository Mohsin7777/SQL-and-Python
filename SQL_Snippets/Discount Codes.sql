



----- 1: Get a list of all Regular Ringback Socs




---------------------------- 1: R Socs

Select SOC into #soc
from STAGE.dbo.STG_RATED_FEATURE
where feature_code = 'RINGBK'
AND EXPIRATION_DATE is null


---------------------------- 2: PPRC Feature_Code

select a.* into #pprc
from STAGE.dbo.STG_PP_RC_RATE a
join #soc b
ON a.SOC=b.soc 
WHERE SUSPENSION_AMOUNT > '0'
AND RATE = 1

select * from #pprc 

---------------------------- 3: Feature Category

select a.* into #chargeinfo
from [STAGE].[dbo].[STG_CHARGE_INFO] a
JOIN #pprc b
on a.FEATURE_CODE=b.feature_code


---------------------------- 4: DISCount Code

select a.* into #disc
from STAGE.dbo.STG_DISCOUNT_CATEGORY a 
join #chargeinfo b
on a.FEATURE_CATEGORY=b.FEATURE_CATEGORY

select * from #disc
where DISC_PLAN_CD
IN ('RS001MN00','RS001MA00')

---------------------------- 5: Discount Code !!

select a.* into #codes
from STAGE.dbo.STG_DISCOUNT_PLAN a
join #disc b
on a.DISCOUNT_CODE=b.disc_plan_cd

select *
 from #codes
where exp_date is null
AND DISC_QTY_TYPE = 'D'
AND FLAT_AMT = '1'


  select COUNT(distinct(cast(ban as varchar(20))+subscriber_no))
  from [STAGE].[dbo].[STG_BAN_DISCOUNT]
  where discount_code IN ('RS001MN00','RS001MA00')
  AND EFFECTIVE_DATE > '2011-11-08'



-------------76,084 people with discount post Nov 8

  select COUNT(distinct(cast(ban as varchar(20))+subscriber_no))
  from [STAGE].[dbo].[STG_BAN_DISCOUNT]
  where discount_code IN ('RS001MN00','RS001MA00')

select *
from [STAGE].[dbo].[STG_BAN_DISCOUNT]



