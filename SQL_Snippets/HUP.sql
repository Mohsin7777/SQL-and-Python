
------------- ALL HUPS IN TIME PERIOD

SELECT DISTINCT H.BAN, H.SUBSCRIBER_NO, H.DEALER_CODE, H.HUP_STATUS_DATE
into #hup
FROM STAGE.dbo.STG_HUP_UPGRADE_HISTORY H
WHERE H.PRODUCT_TYPE = 'C'
AND H.HUP_TYPE != 'S' AND H.HUP_TYPE != 'O'
AND H.HUP_REQUEST_STATUS = 'A'
AND H.NEW_EQ_TYPE != 'G'
AND H.CANCEL_DATE IS NULL
AND H.HUP_STATUS_DATE between '01/01/2012' and '01/31/2012'