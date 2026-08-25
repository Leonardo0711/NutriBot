-- WhatsApp usernames can replace the E.164 number with a Twilio BSUID.
-- Twilio documents a maximum channel address length of 140 characters.
ALTER TABLE usuarios
    ALTER COLUMN numero_whatsapp TYPE varchar(140),
    ALTER COLUMN whatsapp_jid TYPE varchar(160);

ALTER TABLE outgoing_messages
    ALTER COLUMN phone TYPE varchar(140);

COMMENT ON COLUMN usuarios.numero_whatsapp IS
    'Identificador de WhatsApp: telefono E.164 limpio o BSUID de Twilio.';
COMMENT ON COLUMN usuarios.whatsapp_jid IS
    'Identificador interno completo derivado del telefono o BSUID de WhatsApp.';
COMMENT ON COLUMN outgoing_messages.phone IS
    'Destino de WhatsApp: telefono E.164 limpio o BSUID de Twilio.';
