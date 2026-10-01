"""Exploratory profiling of the CSCP dataset. Usage: python scripts/profile_data.py data/raw/interviewtestdataset.csv"""
import sys
# ---- p1 ----
import pandas as pd
f=sys.argv[1] if len(sys.argv)>1 else "data/raw/interviewtestdataset.csv"
df=pd.read_csv(f,dtype=str,keep_default_na=False,na_values=[""])
print(df.shape); print(list(df.columns))
print(df.head(3).T)
print("nulls:\n",df.isna().sum())
print("nunique:\n",df.nunique())
print("full dup rows",df.duplicated().sum())
print("dup (CDPHId,CSFId,ChemicalId)",df.duplicated(["CDPHId","CSFId","ChemicalId"]).sum())
print("ChemicalId unique per row?", df.ChemicalId.nunique())
vc=df.CasNumber.value_counts(dropna=False); print(vc.head(10)); print("TiO2 share", (df.CasNumber=="13463-67-7").mean())
n2c=df.dropna(subset=["CasNumber"]).groupby("ChemicalName").CasNumber.nunique(); print("names>1 CAS",(n2c>1).sum()); print(n2c[n2c>1])
c2n=df.groupby("CasNumber").ChemicalName.nunique(); print("CAS>1 name",(c2n>1).sum())
for c in c2n[c2n>1].index: print(c, df[df.CasNumber==c].ChemicalName.unique())
print("null CAS by name:\n",df[df.CasNumber.isna()].ChemicalName.value_counts().head(10))
# ---- p2 ----
import pandas as pd, re
f=sys.argv[1] if len(sys.argv)>1 else "data/raw/interviewtestdataset.csv"
df=pd.read_csv(f,dtype=str,keep_default_na=False,na_values=[""])
raw=pd.read_csv(f,dtype=str)
print("BrandName null default-na:",raw.BrandName.isna().sum(), "literal strings:", df.BrandName[raw.BrandName.isna() & df.BrandName.notna()].unique())
print("TiO2 CAS variants:", df[df.ChemicalName=="Titanium dioxide"].CasNumber.value_counts(dropna=False).to_dict())
bad=df.CasNumber.dropna(); print("CAS not matching pattern:", bad[~bad.str.fullmatch(r"\d{2,7}-\d{2}-\d")].value_counts().to_dict())
print("CasId per CasNumber>1:", df.groupby("CasNumber").CasId.nunique().pipe(lambda s:s[s>1]).to_dict())
print("CompanyName w/ >1 CompanyId:", df.groupby("CompanyName").CompanyId.nunique().pipe(lambda s:(s>1).sum()))
print("CompanyId w/ >1 name:", df.groupby("CompanyId").CompanyName.nunique().pipe(lambda s:(s>1).sum()))
print("SubCategory >1 Id:", df.groupby("SubCategory").SubCategoryId.nunique().pipe(lambda s:s[s>1]).to_dict())
print("SubCategory trailing ws:", (df.SubCategory!=df.SubCategory.str.strip()).sum(), "distinct stripped", df.SubCategory.str.strip().nunique())
print("subcat in >1 primary:", df.assign(s=df.SubCategory.str.strip()).groupby("s").PrimaryCategory.nunique().pipe(lambda s:(s>1).sum()))
print(df.PrimaryCategory.value_counts().to_dict())
for c in ["CompanyName","BrandName","ProductName"]:
    s=df[c].dropna(); print(c,"ws:",(s!=s.str.strip()).sum(),"distinct",s.nunique(),"casefold+strip",s.str.strip().str.casefold().nunique())
print("brands under >1 company:", df.groupby(df.BrandName.str.casefold()).CompanyName.nunique().pipe(lambda s:(s>1).sum()))
print("CDPHId: >1 product name", df.groupby("CDPHId").ProductName.nunique().pipe(lambda s:(s>1).sum()),
 ">1 company",df.groupby("CDPHId").CompanyId.nunique().pipe(lambda s:(s>1).sum()),
 ">1 subcat",df.groupby("CDPHId").SubCategoryId.nunique().pipe(lambda s:(s>1).sum()),
 ">1 InitialDate",df.groupby("CDPHId").InitialDateReported.nunique().pipe(lambda s:(s>1).sum()),
 ">1 Discontinued",df.groupby("CDPHId").DiscontinuedDate.nunique().pipe(lambda s:(s>1).sum()))
print("ChemicalId -> >1 CDPHId", df.groupby("ChemicalId").CDPHId.nunique().pipe(lambda s:(s>1).sum()), ">1 CasId", df.groupby("ChemicalId").CasId.nunique().pipe(lambda s:(s>1).sum()))
print("rows per ChemicalId", df.ChemicalId.value_counts().describe().to_dict())
# ChemicalCount vs actual
cc=df.groupby("CDPHId").agg(cc=("ChemicalCount",lambda s:s.astype(int).max()), n=("CasId","nunique"))
print("ChemicalCount values",df.ChemicalCount.value_counts().to_dict()); print("CC==distinct CasId:", (cc.cc==cc.n).mean())
# ---- p3 ----
import pandas as pd
f=sys.argv[1] if len(sys.argv)>1 else "data/raw/interviewtestdataset.csv"
df=pd.read_csv(f,dtype=str,keep_default_na=False,na_values=[""])
D=["InitialDateReported","MostRecentDateReported","DiscontinuedDate","ChemicalCreatedAt","ChemicalUpdatedAt","ChemicalDateRemoved"]
for c in D:
    s=df[c].dropna(); p=pd.to_datetime(s,format="%m/%d/%Y",errors="coerce")
    print(f"{c}: n={len(s)} unparsable={p.isna().sum()} min={p.min().date()} max={p.max().date()} >2021={(p.dt.year>2021).sum()} <2009={(p.dt.year<2009).sum()}")
    print("   by year:", p.dt.year.value_counts().sort_index().to_dict())
p={c:pd.to_datetime(df[c],format="%m/%d/%Y",errors="coerce") for c in D}
print("Recent<Initial", (p["MostRecentDateReported"]<p["InitialDateReported"]).sum())
print("Discontinued<Initial", (p["DiscontinuedDate"]<p["InitialDateReported"]).sum())
print("Removed<Created", (p["ChemicalDateRemoved"]<p["ChemicalCreatedAt"]).sum())
print("Updated<Created", (p["ChemicalUpdatedAt"]<p["ChemicalCreatedAt"]).sum())
# ChemicalId->>1 CDPHId examples
g=df.groupby("ChemicalId").CDPHId.nunique(); ex=g[g>1].index[:2]
print(df[df.ChemicalId.isin(ex)][["CDPHId","CSFId","ChemicalId","ChemicalName","ProductName"]].head(6))
# dup triples: what differs?
k=["CDPHId","CSFId","ChemicalId"]
d=df[df.duplicated(k,keep=False)]
diffcols={c:(d.groupby(k,dropna=False)[c].nunique(dropna=False)>1).sum() for c in df.columns if c not in k}
print("dup-triple groups:",d.groupby(k,dropna=False).ngroups,"differing cols:",{a:b for a,b in diffcols.items() if b})
print("dup triples with null CSFId:", d.CSFId.isna().mean())
print("CSF per CDPHId desc:", df.groupby("CDPHId").CSFId.nunique().describe().to_dict())
print("CSFId null but CSF not:",(df.CSFId.isna()&df.CSF.notna()).sum())
print("ChemicalCount 0 rows sample:", df[df.ChemicalCount=="0"][["ChemicalName","ChemicalDateRemoved"]].notna().mean().to_dict())
# products per chemical canonical
print("products per ChemicalName (top):", df.groupby("ChemicalName").CDPHId.nunique().sort_values(ascending=False).head(8).to_dict())
print("TiO2 share of products:", df[df.ChemicalName.str.contains("Titanium")].CDPHId.nunique()/df.CDPHId.nunique())
print("all chem names:"); print(sorted(df.ChemicalName.unique()))
# ---- p4 ----
import pandas as pd
f=sys.argv[1] if len(sys.argv)>1 else "data/raw/interviewtestdataset.csv"
df=pd.read_csv(f,dtype=str,keep_default_na=False,na_values=[""])
a=df[df.CasNumber.str.strip()=="75-07-0"]; print("75-07-0 rows",len(a),"products",a.CDPHId.nunique(), a.ChemicalName.unique())
print("CasId->names", df.groupby("CasId").ChemicalName.nunique().max(), " CasId->CasNumber>1", (df.groupby("CasId").CasNumber.nunique()>1).sum())
print("name->CasId>1", (df.groupby("ChemicalName").CasId.nunique()>1).sum())
r=df[pd.to_datetime(df.ChemicalDateRemoved,format="%m/%d/%Y",errors="coerce").dt.year>2100]
print("2103/2104 removed: dates",r.ChemicalDateRemoved.unique()[:5],"companies",r.CompanyName.nunique(), r.CompanyName.value_counts().head(3).to_dict())
cn=df.groupby("CompanyName").CompanyId.nunique(); print("same name diff ids e.g.", cn[cn>1].index[:4].tolist())
print("Trade Secret rows", (df.ChemicalName=="Trade Secret").sum(), "CAS", df[df.ChemicalName=="Trade Secret"].CasNumber.unique()[:5])
print("CAS '0' names", df[df.CasNumber=="0"].ChemicalName.value_counts().head(5).to_dict())
b=df.BrandName.dropna().str.strip(); print("brand casing ex:", df[df.BrandName.str.strip().str.casefold()=="sally hansen"].BrandName.unique() if (b.str.casefold()=="sally hansen").any() else b.str.casefold().value_counts().head(3))
print("top companies by products", df.groupby("CompanyName").CDPHId.nunique().sort_values(ascending=False).head(5).to_dict())
print("discontinued products", df[df.DiscontinuedDate.notna()].CDPHId.nunique(), "removed product-chem rows", df.ChemicalDateRemoved.notna().sum())
print("ProductName ambiguity: names w/ >1 CDPHId", (df.groupby(df.ProductName.str.strip().str.casefold()).CDPHId.nunique()>1).sum())
