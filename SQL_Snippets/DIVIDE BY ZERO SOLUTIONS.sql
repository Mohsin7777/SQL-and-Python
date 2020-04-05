SET ARITHABORT OFF
SET ANSI_WARNINGS OFF

----- THEN TYPE QUERY




--#2:


SELECT A.END_DATE
     , A.FRANCHISE
     ,  B.CHANNEL_TYPE_DESC
     , isnull((sum([ACTS_ATTACH]))/(sum([TOTAL_ACTS])) ,0) as "ACTIVATION ATTACH %"
     ,isnull((sum([HUP_ATTACH]))/(sum([TOTAL_HUP])) ,0) as "HUP ATTACH %"
     ,isnull((sum([PPC_ATTACH]))/(sum([TOTAL_PPC])) ,0) as "PPC %"