


---- Video Downloads

Select count(*)
from entertainment.dbo.stg_cms_purchase_trans
WHERE month (transaction_datetime)=4
AND year (transaction_datetime)=2011
AND content_type IN ('MVIDDL','SVIDDL')
AND transaction_status_indicator = 'Successful'








-- Ringtone Subscriptions REDEMPTIONS----


Select count(*)
from entertainment.dbo.steg_cms_purchase_trans
WHERE month (transaction_datetime)=1
AND year (transaction_datetime)=2011
AND content_type = '30'
AND transaction_status_indicator = 'Successful'
AND shortcode is NULL



-- Ringtone Subscriptions Purchased ----

Select count(*)
from entertainment.dbo.stg_cms_purchase_trans
WHERE month (transaction_datetime)=1
AND year (transaction_datetime)=2011
AND content_type = '30'
AND transaction_status_indicator = 'Successful'
AND shortcode is NULL






