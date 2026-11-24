// Migration 0002 — Case-law v1: additional constraints, indexes, and seed reference nodes.
// All statements are idempotent (IF NOT EXISTS / MERGE).
// Never edit this file after it has been applied — write 0003 instead.

// ---------- additional uniqueness constraints ----------
CREATE CONSTRAINT article_key      IF NOT EXISTS FOR (n:Article)         REQUIRE n.article_key IS UNIQUE;
CREATE CONSTRAINT rule_key         IF NOT EXISTS FOR (n:Rule)             REQUIRE n.rule_key IS UNIQUE;
CREATE CONSTRAINT definition_key   IF NOT EXISTS FOR (n:Definition)       REQUIRE n.definition_key IS UNIQUE;
CREATE CONSTRAINT claim_type_code  IF NOT EXISTS FOR (n:ClaimType)        REQUIRE n.code IS UNIQUE;
CREATE CONSTRAINT outcome_code     IF NOT EXISTS FOR (n:Outcome)          REQUIRE n.code IS UNIQUE;
CREATE CONSTRAINT award_key        IF NOT EXISTS FOR (n:Award)            REQUIRE n.award_key IS UNIQUE;
CREATE CONSTRAINT contract_elem    IF NOT EXISTS FOR (n:ContractElement)  REQUIRE n.code IS UNIQUE;
CREATE CONSTRAINT gate_condition   IF NOT EXISTS FOR (n:GateCondition)    REQUIRE n.code IS UNIQUE;
CREATE CONSTRAINT station_key      IF NOT EXISTS FOR (n:Station)          REQUIRE n.station_key IS UNIQUE;
CREATE CONSTRAINT party_type_code  IF NOT EXISTS FOR (n:PartyType)        REQUIRE n.code IS UNIQUE;
CREATE CONSTRAINT lower_court_key  IF NOT EXISTS FOR (n:LowerCourtDecision) REQUIRE n.lcd_key IS UNIQUE;

// ---------- additional indexes ----------
CREATE INDEX para_case_no     IF NOT EXISTS FOR (n:Paragraph)  ON (n.case_key, n.para_no);
CREATE INDEX provision_statute IF NOT EXISTS FOR (n:Provision) ON (n.statute_key);
CREATE INDEX award_type        IF NOT EXISTS FOR (n:Award)     ON (n.type);
CREATE INDEX principle_status  IF NOT EXISTS FOR (n:Principle) ON (n.status);
CREATE INDEX case_in_corpus2   IF NOT EXISTS FOR (n:Case)      ON (n.in_corpus, n.court_code);

// ==========================================================================
// Seed: Courts and BINDS hierarchy
// SC → CoA → {HC, ELC, ELRC} → {SCC, MC}
// ==========================================================================
MERGE (sc:Court {code: "KESC"})   SET sc.name = "Supreme Court of Kenya",         sc.rank = 1;
MERGE (coa:Court {code: "KECA"})  SET coa.name = "Court of Appeal of Kenya",      coa.rank = 2;
MERGE (hc:Court {code: "KEHC"})   SET hc.name = "High Court of Kenya",            hc.rank = 3;
MERGE (elc:Court {code: "KEELC"}) SET elc.name = "Environment and Land Court",    elc.rank = 3;
MERGE (elrc:Court {code: "KEELRC"}) SET elrc.name = "Employment and Labour Relations Court", elrc.rank = 3;
MERGE (scc:Court {code: "SCC"})   SET scc.name = "Small Claims Court",            scc.rank = 4;
MERGE (mc:Court {code: "KEMC"})   SET mc.name = "Magistrate Court of Kenya",      mc.rank = 4;
MERGE (eaca:Court {code: "EACA"}) SET eaca.name = "East Africa Court of Appeal",  eaca.rank = 9;
MERGE (foreign:Court {code: "FOREIGN"}) SET foreign.name = "Foreign Court",       foreign.rank = 9;

MERGE (sc)-[:BINDS]->(coa);
MERGE (sc)-[:BINDS]->(hc);
MERGE (sc)-[:BINDS]->(elc);
MERGE (sc)-[:BINDS]->(elrc);
MERGE (sc)-[:BINDS]->(scc);
MERGE (sc)-[:BINDS]->(mc);
MERGE (coa)-[:BINDS]->(hc);
MERGE (coa)-[:BINDS]->(elc);
MERGE (coa)-[:BINDS]->(elrc);
MERGE (coa)-[:BINDS]->(scc);
MERGE (coa)-[:BINDS]->(mc);
MERGE (hc)-[:BINDS]->(scc);
MERGE (hc)-[:BINDS]->(mc);
MERGE (elc)-[:BINDS]->(scc);
MERGE (elc)-[:BINDS]->(mc);
MERGE (elrc)-[:BINDS]->(scc);
MERGE (elrc)-[:BINDS]->(mc);

// ==========================================================================
// Seed: ContractElement A–K
// ==========================================================================
MERGE (ce_a:ContractElement {code: "A"}) SET ce_a.name = "Offer";
MERGE (ce_b:ContractElement {code: "B"}) SET ce_b.name = "Acceptance";
MERGE (ce_c:ContractElement {code: "C"}) SET ce_c.name = "Consideration";
MERGE (ce_d:ContractElement {code: "D"}) SET ce_d.name = "Intention to create legal relations";
MERGE (ce_e:ContractElement {code: "E"}) SET ce_e.name = "Capacity";
MERGE (ce_f:ContractElement {code: "F"}) SET ce_f.name = "Legality of object";
MERGE (ce_g:ContractElement {code: "G"}) SET ce_g.name = "Certainty of terms";
MERGE (ce_h:ContractElement {code: "H"}) SET ce_h.name = "Performance / breach";
MERGE (ce_i:ContractElement {code: "I"}) SET ce_i.name = "Damages";
MERGE (ce_j:ContractElement {code: "J"}) SET ce_j.name = "Specific relief";
MERGE (ce_k:ContractElement {code: "K"}) SET ce_k.name = "Limitation";

// ==========================================================================
// Seed: GateCondition + stub Provision nodes + GROUNDED_IN edges
// ==========================================================================
MERGE (prov_s106b:Statute {statute_key: "evidence-act"})
  SET prov_s106b.name = "Evidence Act", prov_s106b.short_name = "EA";

MERGE (p4a:Provision {provision_key: "evidence-act/s106B/4/a"})
  SET p4a.statute_key = "evidence-act", p4a.section = "s.106B(4)(a)";
MERGE (p4b:Provision {provision_key: "evidence-act/s106B/4/b"})
  SET p4b.statute_key = "evidence-act", p4b.section = "s.106B(4)(b)";
MERGE (p2a:Provision {provision_key: "evidence-act/s106B/2/a"})
  SET p2a.statute_key = "evidence-act", p2a.section = "s.106B(2)(a)";
MERGE (p2b:Provision {provision_key: "evidence-act/s106B/2/b"})
  SET p2b.statute_key = "evidence-act", p2b.section = "s.106B(2)(b)";
MERGE (p2c:Provision {provision_key: "evidence-act/s106B/2/c"})
  SET p2c.statute_key = "evidence-act", p2c.section = "s.106B(2)(c)";
MERGE (p2d:Provision {provision_key: "evidence-act/s106B/2/d"})
  SET p2d.statute_key = "evidence-act", p2d.section = "s.106B(2)(d)";
MERGE (p4d:Provision {provision_key: "evidence-act/s106B/4/d"})
  SET p4d.statute_key = "evidence-act", p4d.section = "s.106B(4)(d)";
MERGE (p2:Provision  {provision_key: "evidence-act/s106B/2"})
  SET p2.statute_key  = "evidence-act", p2.section = "s.106B(2)";
MERGE (s13_5:Provision {provision_key: "scc-act/s13/5"})
  SET s13_5.statute_key = "scc-act", s13_5.section = "s.13(5)";

MERGE (scc_act:Statute {statute_key: "scc-act"})
  SET scc_act.name = "Small Claims Court Act", scc_act.short_name = "SCC Act";

MERGE (gc4a:GateCondition {code: "4a"}) SET gc4a.label = "s.106B(4)(a) — made by responsible official";
MERGE (gc4b:GateCondition {code: "4b"}) SET gc4b.label = "s.106B(4)(b) — identifies the computer";
MERGE (gc2a:GateCondition {code: "2a"}) SET gc2a.label = "s.106B(2)(a) — regularly used";
MERGE (gc2b:GateCondition {code: "2b"}) SET gc2b.label = "s.106B(2)(b) — in proper working order";
MERGE (gc2c:GateCondition {code: "2c"}) SET gc2c.label = "s.106B(2)(c) — stored in ordinary course";
MERGE (gc2d:GateCondition {code: "2d"}) SET gc2d.label = "s.106B(2)(d) — information reproduced correctly";
MERGE (gc4d:GateCondition {code: "4d"}) SET gc4d.label = "s.106B(4)(d) — certified by responsible official";
MERGE (gcINT:GateCondition {code: "INTEGRITY"}) SET gcINT.label = "Device integrity (s.106B(2) general)";

MERGE (gc4a)-[:GROUNDED_IN]->(p4a);
MERGE (gc4b)-[:GROUNDED_IN]->(p4b);
MERGE (gc2a)-[:GROUNDED_IN]->(p2a);
MERGE (gc2b)-[:GROUNDED_IN]->(p2b);
MERGE (gc2c)-[:GROUNDED_IN]->(p2c);
MERGE (gc2d)-[:GROUNDED_IN]->(p2d);
MERGE (gc4d)-[:GROUNDED_IN]->(p4d);
MERGE (gcINT)-[:GROUNDED_IN]->(p2);

// ==========================================================================
// Seed: EvidenceType
// ==========================================================================
MERGE (et_wa:EvidenceType {code: "whatsapp"})    SET et_wa.label = "WhatsApp message";
MERGE (et_sm:EvidenceType {code: "sms"})         SET et_sm.label = "SMS message";
MERGE (et_mp:EvidenceType {code: "mpesa"})       SET et_mp.label = "M-Pesa statement / receipt";
MERGE (et_em:EvidenceType {code: "email"})       SET et_em.label = "Email";
MERGE (et_ph:EvidenceType {code: "photo"})       SET et_ph.label = "Photograph";
MERGE (et_cc:EvidenceType {code: "cctv"})        SET et_cc.label = "CCTV footage";
MERGE (et_dv:EvidenceType {code: "dvd"})         SET et_dv.label = "DVD / disc recording";
MERGE (et_sp:EvidenceType {code: "social_post"}) SET et_sp.label = "Social media post";
MERGE (et_dc:EvidenceType {code: "document"})    SET et_dc.label = "Electronic document";

// ==========================================================================
// Seed: Outcome codes
// ==========================================================================
MERGE (o1:Outcome {code: "ALLOWED"})            SET o1.label = "Appeal/Application allowed";
MERGE (o2:Outcome {code: "DISMISSED"})          SET o2.label = "Dismissed";
MERGE (o3:Outcome {code: "STRUCK_OUT"})         SET o3.label = "Struck out";
MERGE (o4:Outcome {code: "SET_ASIDE_REFUSED"})  SET o4.label = "Set aside refused";
MERGE (o5:Outcome {code: "SET_ASIDE_GRANTED"})  SET o5.label = "Set aside granted";
MERGE (o6:Outcome {code: "OBJECTION_UPHELD"})   SET o6.label = "Objection upheld";
MERGE (o7:Outcome {code: "OBJECTION_DISMISSED"}) SET o7.label = "Objection dismissed";
MERGE (o8:Outcome {code: "PARTLY_ALLOWED"})     SET o8.label = "Partly allowed";
MERGE (o9:Outcome {code: "GRANTED"})            SET o9.label = "Granted";
MERGE (o10:Outcome {code: "REFUSED"})           SET o10.label = "Refused";

// ==========================================================================
// Seed: ClaimType
// ==========================================================================
MERGE (ct1:ClaimType {code: "money_lent"})            SET ct1.label = "Money lent";
MERGE (ct2:ClaimType {code: "sale_goods_services"})  SET ct2.label = "Sale of goods / services";
MERGE (ct3:ClaimType {code: "property_damage_tort"}) SET ct3.label = "Property damage (tort)";
MERGE (ct4:ClaimType {code: "personal_injury"})      SET ct4.label = "Personal injury";
MERGE (ct5:ClaimType {code: "subrogation_recovery"}) SET ct5.label = "Subrogation recovery";
MERGE (ct6:ClaimType {code: "employment_founded"})   SET ct6.label = "Employment founded claim";
MERGE (ct7:ClaimType {code: "defamation"})           SET ct7.label = "Defamation";
MERGE (ct8:ClaimType {code: "bailment"})             SET ct8.label = "Bailment";
MERGE (ct9:ClaimType {code: "other"})                SET ct9.label = "Other";

MERGE (emp_ct:ClaimType {code: "employment_founded"})
MERGE (s13_5p:Provision {provision_key: "scc-act/s13/5"})
MERGE (emp_ct)-[:EXCLUDED_BY]->(s13_5p);

// ==========================================================================
// Seed: PartyType
// ==========================================================================
MERGE (pt1:PartyType {code: "individual"})  SET pt1.label = "Individual";
MERGE (pt2:PartyType {code: "company"})    SET pt2.label = "Company";
MERGE (pt3:PartyType {code: "insurer"})    SET pt3.label = "Insurance company";
MERGE (pt4:PartyType {code: "employer"})   SET pt4.label = "Employer";
MERGE (pt5:PartyType {code: "government"}) SET pt5.label = "Government / public body";
