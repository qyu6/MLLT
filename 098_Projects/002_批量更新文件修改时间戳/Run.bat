chcp 65001
setlocal enabledelayedexpansion

for /r "C:\Users\Tony\Downloads" %%i in (*) do (
	echo 1. %%~ti
	set str=%%~ti
	echo 2. !str!
	set str1=!str:~0,4!
	echo 3. !str1!
	echo 4. %%i
	if !str1! LEQ 2022 (
	touch "%%i" -c
	) else (
	echo 5. not meet touch condition
	)
)
