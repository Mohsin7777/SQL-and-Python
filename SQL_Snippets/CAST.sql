  
  select CAST (MSISDN as varchar(9)) as Sub
		,[Album artist]
		into #temp
		from  [ENTERTAINMENT].[dbo].[Winnepeg_SaleRentReportBundles]
		