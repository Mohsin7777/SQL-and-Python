

UPDATE A
SET A."MIGRATIONS_TO_VP"=B."MIGRATIONS_TO_VP"
FROM [ENTERTAINMENT].[dbo].[RM_VALUEPACK_WEEKLY_NONVP_SOCS]  A
INNER JOIN #VMCLID_MIGRATIONS_TO_VP B
ON A.WEEK_NO=B.WEEK_NO
AND A.FRANCHISE=B.FRANCHISE
AND A.END_DATE=B.END_DATE
AND A.RATE=B.RATE
AND A.[SOC_REV_CODE]=B.SOC_TYPE
