select  a.*, CAST(DateName( month, DateAdd( month , b.MONTH_DATE , 0 ) - 1 ) AS VARCHAR(20)) + ', ' + CAST(B.YEAR_DATE AS VARCHAR(20)) AS "DATE"
  FROM [REPORTINGMARTS].[dbo].[RM_ENTERTAINMENT_DEVICES_MONTHLY] a
    JOIN STAGE.dbo.STG_TIME_DIM_MONTHLY b
      on a.Index_no=b.INDEX_NO
  and a.DEVICE_TYPE_DESC = 'android'
  and index_no = @index
  order by  index_no desc, MONTHLY_TRANSACTIONS desc
  
  