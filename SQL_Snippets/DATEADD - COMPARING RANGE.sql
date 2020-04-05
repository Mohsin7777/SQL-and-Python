
SELECT A.*,
ATTACH = CASE
WHEN B.BAN IS NULL 
THEN 0
ELSE 1
END
,B.SOC
INTO #ATTACH_15
FROM #ACT_SWAPPED A
LEFT OUTER JOIN ##SERVICE_VALUEPACKS B
ON A.BAN=B.BAN
AND A.SUBSCRIBER_NO=B.SUBSCRIBER_NO
WHERE (A.INIT_ACTIVATION_DATE  BETWEEN B.EFFECTIVE_DATE
AND DATEADD(DAY,+15,B.EFFECTIVE_DATE))


SELECT SUM([ATTACH])
FROM #ATTACH_15
