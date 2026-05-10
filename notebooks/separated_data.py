import pandas as pd

# Base folder
base_path = r"C:\Users\A\Downloads"

# Load all datasets
customers = pd.read_csv(rf"{base_path}\customers-10000.csv")
leads = pd.read_csv(rf"{base_path}\leads-10000.csv")
organizations = pd.read_csv(rf"{base_path}\organizations-10000.csv")
people = pd.read_csv(rf"{base_path}\people-10000.csv")
products = pd.read_csv(rf"{base_path}\products-10000.csv")

# Create merged Excel file
output_file = rf"{base_path}\merged_data.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    customers.to_excel(writer, sheet_name="Customers", index=False)
    leads.to_excel(writer, sheet_name="Leads", index=False)
    organizations.to_excel(writer, sheet_name="Organizations", index=False)
    people.to_excel(writer, sheet_name="People", index=False)
    products.to_excel(writer, sheet_name="Products", index=False)

print(f"Done! File created at:\n{output_file}")