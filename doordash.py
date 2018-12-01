import sys
import doordash_part2 as dp2

#following libraries are needed
# pandas >> python -m pip install pandas 
# sklean >> python -m pip install sklearn

#instruction to run this file
#>>python 

def main():
	if len(sys.argv) > 1:
		json_file = sys.argv[1]
	else:
		print("enter json_file as input")
	dp2.predict_delivery_time(json_file)
		
if __name__ == "__main__":
    main()

