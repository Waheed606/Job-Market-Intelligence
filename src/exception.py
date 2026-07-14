import sys

def error_message_details(error,error_detials:sys):
    _,_,exc_tb=error_detials.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message=f'Error occur in python script name {file_name} line number {exc_tb.tb_lineno} error message {str(error)}'


class CustomeExpection(Exception):

    def __init___(self,error_message,error_details:sys):
        super.__init__(self,error_message)
        self.error_message=error_message_details(error_message,error_details)

    def __str__(self):
        return self.error_message

# This code is used for testing
# if __name__=='__main__':
#     try:
#         p=2/0
#     except Exception as e:
#         raise CustomeExpection(e,sys)
