


select 
brand = CASE 
when brand = 'Other' 
then 'Rogers' else brand 
end 
,[promotional_code] = CASE
when promotional_code IS NULL 
THEN 'N'
ELSE 'Y'
END


GROUP BY BRAND 