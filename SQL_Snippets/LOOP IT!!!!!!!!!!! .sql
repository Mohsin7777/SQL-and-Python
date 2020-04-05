
DECLARE @WEEK AS NUMERIC(4,0)
DECLARE @START AS DATETIME
DECLARE @END AS DATETIME

set @week = (select MAX ([WEEK])-3       ------------------ TAKE AWAY (-#) FOR PAST WEEK
from [STAGE].[dbo].[STG_TIME_DIM_WEEKLY]
where [ENDDATE] +.9999999 <= GETDATE())

while @WEEK >= 145                  ------------------- (START WEEK)
begin

select @week


SET @WEEK = @WEEK - 1
END
GO
