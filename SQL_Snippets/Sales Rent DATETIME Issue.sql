 
 
 ------- Remember to filter OUT [Product Type 33] in reports ('33' is playcounts and is aggregated in salesrent)
  
 
-------- !!! Don't open CSV file in Excel - Just upload directly on the server or Excel will screw it up

--- WHILE IMPORTING, SELECT DT_DATE FORMAT, FOR DATE OF SALE!!!


  
  ---- TO CAST (IN CASE YOU DIDNT SELECT THE FORMAT)  
  
  
  
  
  
  
  
  
  select CAST([Date of Sale] AS DATETIME) as [Date of Sale]
   ,[Label]
      ,[Genre]
      ,[Album Artist]
      ,[Album Title]
      ,[Track Artist]
      ,[Track Title]
      ,[Physical UPC]
      ,[Digital UPC]
      ,[ISRC]
      ,[GRID]
      ,[Proprietary Release ID]
      ,[Proprietary Track ID]
      ,[SoundScan ID]
      ,[CSI Track ID]
      ,[Disc Number]
      ,[Track Number]
      ,[Product Type]
      ,[Delivery]
      ,[Retail Price]
      ,[Promotion]
      ,[Bundle]
      ,[Brand]
      ,[MSISDN]
      ,[Handset]
      ,[Adora Content ID]
      ,[Plays]
      ,[Type]
      ,[Price Group]
	
into #temp1 
from [ENTERTAINMENT].[dbo].[Adhoc_Nov11-Jan19-SaleRentReport]











----------- Step #2 ------- CREATE Table with Proper Formay

CREATE TABLE entertainment.dbo.[Adhoc_DATEFIXED_Nov11-Jan19-SaleRentReport]
(
	DATE_OF_SALE DATETIME,
	[Label] VARCHAR(500)
      ,[Genre]  VARCHAR(500)
      ,[Album Artist] VARCHAR(500)
      ,[Album Title] VARCHAR(500)
      ,[Track Artist] VARCHAR(500)
      ,[Track Title] VARCHAR(500)
      ,[Physical UPC] VARCHAR(500)
      ,[Digital UPC] VARCHAR(500)
      ,[ISRC] VARCHAR(500)
      ,[GRID] VARCHAR(500)
      ,[Proprietary Release ID] VARCHAR(500)
      ,[Proprietary Track ID] VARCHAR(500)
      ,[SoundScan ID] VARCHAR(500)
      ,[CSI Track ID] VARCHAR(500)
      ,[Disc Number] VARCHAR(500)
      ,[Track Number] VARCHAR(500)
      ,[Product Type] VARCHAR(500)
      ,[Delivery] VARCHAR(500)
      ,[Retail Price] Numeric (18,2)
      ,[Promotion] VARCHAR(500)
      ,[Bundle] VARCHAR(500)
      ,[Brand] VARCHAR(500)
      ,[MSISDN] VARCHAR(500)
      ,[Handset] VARCHAR(500)
      ,[Adora Content ID] VARCHAR(500)
      ,[Plays] VARCHAR(500)
      ,[Type] VARCHAR(500)
      ,[Price Group] VARCHAR(500))
	
INSERT INTO entertainment.dbo.[Adhoc_DATEFIXED_Nov11-Jan19-SaleRentReport]
(
	DATE_OF_SALE,
		[Label]
      ,[Genre]
      ,[Album Artist]
      ,[Album Title]
      ,[Track Artist]
      ,[Track Title]
      ,[Physical UPC]
      ,[Digital UPC]
      ,[ISRC]
      ,[GRID]
      ,[Proprietary Release ID]
      ,[Proprietary Track ID]
      ,[SoundScan ID]
      ,[CSI Track ID]
      ,[Disc Number]
      ,[Track Number]
      ,[Product Type]
      ,[Delivery]
      ,[Retail Price]
      ,[Promotion]
      ,[Bundle]
      ,[Brand]
      ,[MSISDN]
      ,[Handset]
      ,[Adora Content ID]
      ,[Plays]
      ,[Type]
      ,[Price Group] )

SELECT 
	[Date of Sale],
		[Label]
      ,[Genre]
      ,[Album Artist]
      ,[Album Title]
      ,[Track Artist]
      ,[Track Title]
      ,[Physical UPC]
      ,[Digital UPC]
      ,[ISRC]
      ,[GRID]
      ,[Proprietary Release ID]
      ,[Proprietary Track ID]
      ,[SoundScan ID]
      ,[CSI Track ID]
      ,[Disc Number]
      ,[Track Number]
      ,[Product Type]
      ,[Delivery]
      ,[Retail Price]
      ,[Promotion]
      ,[Bundle]
      ,[Brand]
      ,[MSISDN]
      ,[Handset]
      ,[Adora Content ID]
      ,[Plays]
      ,[Type]
      ,[Price Group]

FROM #temp1





  

