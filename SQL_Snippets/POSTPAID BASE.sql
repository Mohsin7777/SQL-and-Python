

------post paid is 1 for rogers and 11 or 5 for fido??????????????

select SUBSCRIBER_NO, customer_ban, franchise_tp, PRICE_PLAN  into #temp
 from STAGE.dbo.STG_SUBSCRIBER
 where SUB_STATUS in ('A','S')
 
 select a.*, pplan_series_type, PPLAN_SERIES_DESC
 from #temp a inner join STAGE.dbo.STG_PRICE_PLAN_GROUP b
 on a.price_plan = b.SOC

select COUNT (